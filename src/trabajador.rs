//! El trabajo de minar: un hilo conduce la GPU y otro construye las shares.
//!
//! Quien recibe los trabajos fija `vigente` en cuanto llega uno, antes de que el motor acabe su lote, y las shares
//! sólo salen si su revisión sigue siendo la vigente. La revisión 0 significa pausa.
use crate::share::Constructor;
use crate::{
    cota_penalizada, job_key, semilla_a_de, Candidato, Motor, Trabajo, Variante, CERT_VERSION,
    HEADER_BYTES,
};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Vueltas por lote con límite: lotes cortos para que un juego no espere mucho a la GPU.
const LOTE_LIMITADO: u32 = 1;
/// Trabajos seguidos con otra versión de certificado que se dejan pasar antes de avisar: uno suelto puede ser un
/// cambio que el pool deshace; tres seguidos son la red, que ha cambiado.
pub const OTRA_VERSION_SEGUIDA: u32 = 3;

/// Cuenta los trabajos seguidos que llegan con una versión de certificado que este motor no mina. Devuelve el aviso
/// la vez que la cuenta llega a `OTRA_VERSION_SEGUIDA`, y sólo esa: sin él, el motor se quedaba parado sin decir nada
/// y quien lo conduce seguía creyendo que minaba.
pub fn otra_version(seguidos: &mut u32, version: u64) -> Option<String> {
    if version == CERT_VERSION {
        *seguidos = 0;
        return None;
    }
    *seguidos = seguidos.saturating_add(1);
    (*seguidos == OTRA_VERSION_SEGUIDA).then(|| {
        format!(
            "Pearl ha cambiado de versión de certificado (ahora la {version}; este programa mina la {CERT_VERSION}).              Hace falta actualizar TokenMiner para seguir minando Pearl."
        )
    })
}

#[derive(Clone)]
pub struct Encargo {
    pub revision: u64,
    pub job_id: String,
    pub cabecera: [u8; HEADER_BYTES],
    pub objetivo: [u8; 32],
    pub cert_version: u64,
}

#[derive(Clone)]
pub enum Orden {
    Trabajo(Encargo),
    Pausa,
    Parar,
}

pub enum Aviso {
    Listo {
        m: u32,
        n: u32,
        mu: u32,
        nu: u32,
        variante: Variante,
        /// Multiprocesadores de esta GPU.
        sm: u32,
    },
    Share {
        revision: u64,
        job_id: String,
        prueba: String,
    },
    Error(String),
}

/// Cuentas que leen los conductores, y el límite que fijan: `limite` es el porcentaje del tiempo que el motor puede
/// tener la GPU ocupada (0 o 100: sin límite) y con `apartado` no lanza nada, sin soltar el trabajo.
#[derive(Default)]
pub struct Contadores {
    pub limite: AtomicU32,
    pub apartado: AtomicBool,
    pub jobs: AtomicU64,
    pub rounds: AtomicU64,
    pub candidates: AtomicU64,
    pub dropped_stale: AtomicU64,
    pub build_errors: AtomicU64,
    pub skipped_cert: AtomicU64,
}

/// 32 B distintos por llamada. No necesitan ser secretos: sólo que no se repita una matriz.
fn semilla_nueva() -> [u8; 32] {
    static CONTADOR: AtomicU64 = AtomicU64::new(0);
    let mut h = blake3::Hasher::new();
    let ahora = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    h.update(&ahora.to_le_bytes());
    h.update(&std::process::id().to_le_bytes());
    h.update(&CONTADOR.fetch_add(1, Ordering::Relaxed).to_le_bytes());
    *h.finalize().as_bytes()
}

struct Pieza {
    revision: u64,
    job_id: String,
    cabecera: [u8; HEADER_BYTES],
    semilla_a0: [u8; 32],
    candidato: Candidato,
}

fn constructor(
    rx: Receiver<Pieza>,
    m: u32,
    n: u32,
    vigente: Arc<AtomicU64>,
    c: Arc<Contadores>,
    avisos: Sender<Aviso>,
) {
    let mut cons = Constructor::default();
    for p in rx {
        if vigente.load(Ordering::SeqCst) != p.revision {
            c.dropped_stale.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let semilla_a = semilla_a_de(&p.semilla_a0, p.candidato.vuelta);
        match cons.prueba(
            &p.cabecera,
            m,
            n,
            &semilla_a,
            p.candidato.fila,
            p.candidato.columna,
        ) {
            Err(e) => {
                c.build_errors.fetch_add(1, Ordering::Relaxed);
                let _ = avisos.send(Aviso::Error(format!("construir: {e}")));
            }
            // Construirla tarda: hay que volver a mirar.
            Ok(_) if vigente.load(Ordering::SeqCst) != p.revision => {
                c.dropped_stale.fetch_add(1, Ordering::Relaxed);
            }
            Ok(prueba) => {
                let _ = avisos.send(Aviso::Share {
                    revision: p.revision,
                    job_id: p.job_id,
                    prueba,
                });
            }
        }
    }
}

/// Una vuelta pequeña en esta GPU contra el código oficial: la mejor tesela de 512 debe pasar el verificador con
/// la dificultad justa para su hash. Una GPU o un driver que calculen otra cosa fallan aquí y no minan shares inválidas.
pub fn autocomprobar(variante: Option<Variante>) -> Result<(), String> {
    autocomprobar_en(0, variante)
}

pub fn autocomprobar_en(dispositivo: i32, variante: Option<Variante>) -> Result<(), String> {
    const CABECERA: [u8; HEADER_BYTES] = [0x5C; HEADER_BYTES];
    let semilla_a0 = [0xA7; 32];
    let (m, n) = (256, 512);
    let mut motor = Motor::en(dispositivo, m, n, 4096, variante)?;
    let job_key = job_key(&CABECERA);
    motor.preparar_trabajo(&Trabajo {
        job_key,
        semilla_b: [0x3D; 32],
        semilla_a0,
        cota: [0xFF; 32],
    })?;
    motor.lote_de_vueltas(1)?;
    let candidatos = motor.vaciar()?;
    let mejor = candidatos
        .iter()
        .min_by_key(|c| {
            let mut h = c.hash;
            h.reverse();
            h
        })
        .ok_or("la GPU no devolvió ninguna tesela")?;
    let prueba = Constructor::default().prueba(
        &CABECERA,
        m,
        n,
        &semilla_a_de(&semilla_a0, mejor.vuelta),
        mejor.fila,
        mejor.columna,
    )?;
    crate::share::verifica_justo(&CABECERA, &prueba, &mejor.hash)
        .map_err(|e| format!("Autocomprobación de Pearl: {e}"))
}

/// Conduce la GPU `dispositivo` hasta `Orden::Parar` o hasta que se cierre el canal de órdenes.
pub fn trabajar(
    dispositivo: i32,
    m: u32,
    n_pedido: u32,
    lote: u32,
    variante: Option<Variante>,
    ordenes: Receiver<Orden>,
    vigente: Arc<AtomicU64>,
    c: Arc<Contadores>,
    avisos: Sender<Aviso>,
) -> Result<(), String> {
    let inicio = autocomprobar_en(dispositivo, variante)
        .and_then(|_| Motor::en(dispositivo, m, n_pedido, 4096, variante));
    let mut motor = match inicio {
        Ok(motor) => motor,
        Err(e) => {
            let _ = avisos.send(Aviso::Error(e.clone()));
            return Err(e);
        }
    };
    let n = motor.n;
    let _ = avisos.send(Aviso::Listo {
        m,
        n,
        mu: motor.mu,
        nu: motor.nu,
        variante: motor.variante,
        sm: motor.sm,
    });
    let (tx, rx) = mpsc::channel::<Pieza>();
    let hilo = {
        let (vigente, c, avisos) = (vigente.clone(), c.clone(), avisos.clone());
        std::thread::spawn(move || constructor(rx, m, n, vigente, c, avisos))
    };
    let mut actual: Option<Encargo> = None;
    let mut semilla_a0 = [0u8; 32];
    let mut de_otra_version = 0;
    'minar: loop {
        // Con trabajo se mira sin esperar; sin él, se duerme hasta la próxima orden.
        let mut ultima = None;
        loop {
            let o = if actual.is_some() {
                match ordenes.try_recv() {
                    Ok(o) => o,
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => break 'minar,
                }
            } else if ultima.is_some() {
                break;
            } else {
                match ordenes.recv_timeout(Duration::from_millis(200)) {
                    Ok(o) => o,
                    Err(RecvTimeoutError::Timeout) => continue,
                    Err(RecvTimeoutError::Disconnected) => break 'minar,
                }
            };
            if matches!(o, Orden::Parar) {
                break 'minar;
            }
            ultima = Some(o);
        }
        match ultima {
            Some(Orden::Trabajo(e)) => {
                motor.vaciar()?; // los candidatos del trabajo anterior no se envían
                actual = None;
                let aviso = otra_version(&mut de_otra_version, e.cert_version);
                if e.cert_version != CERT_VERSION {
                    c.skipped_cert.fetch_add(1, Ordering::Relaxed);
                    if let Some(aviso) = aviso {
                        let _ = avisos.send(Aviso::Error(aviso));
                    }
                    continue;
                }
                let Some(cota) = cota_penalizada(&e.objetivo) else {
                    let _ = avisos.send(Aviso::Error(
                        "objetivo demasiado fácil para esta tesela".into(),
                    ));
                    continue;
                };
                semilla_a0 = semilla_nueva();
                motor.preparar_trabajo(&Trabajo {
                    job_key: job_key(&e.cabecera),
                    semilla_b: semilla_nueva(),
                    semilla_a0,
                    cota,
                })?;
                c.jobs.fetch_add(1, Ordering::Relaxed);
                actual = Some(e);
            }
            Some(Orden::Pausa) => {
                motor.vaciar()?;
                actual = None;
                continue;
            }
            _ => {}
        }
        let Some(e) = &actual else { continue };
        let limite = c.limite.load(Ordering::Relaxed);
        let (vueltas, candidatos, descanso) = if c.apartado.load(Ordering::Relaxed) {
            (0, motor.vaciar()?, Duration::from_millis(100))
        } else if (1..100).contains(&limite) {
            // Con límite no queda nada en cola: se espera el lote entero y la GPU descansa en proporción.
            let (vueltas, t0) = (lote.min(LOTE_LIMITADO), Instant::now());
            let mut candidatos = motor.lote_de_vueltas(vueltas)?;
            candidatos.extend(motor.vaciar()?);
            (vueltas, candidatos, t0.elapsed() * (100 - limite) / limite)
        } else {
            (lote, motor.lote_de_vueltas(lote)?, Duration::ZERO)
        };
        c.rounds.fetch_add(vueltas as u64, Ordering::Relaxed);
        c.candidates
            .fetch_add(candidatos.len() as u64, Ordering::Relaxed);
        for candidato in candidatos {
            let pieza = Pieza {
                revision: e.revision,
                job_id: e.job_id.clone(),
                cabecera: e.cabecera,
                semilla_a0,
                candidato,
            };
            tx.send(pieza).map_err(|e| e.to_string())?;
        }
        std::thread::sleep(descanso);
    }
    motor.vaciar()?;
    vigente.store(0, Ordering::SeqCst);
    drop(tx);
    let _ = hilo.join();
    Ok(())
}

#[cfg(test)]
mod pruebas {
    use super::*;

    #[test]
    fn avisa_una_vez_cuando_la_red_cambia_de_certificado() {
        let mut seguidos = 0;
        // Uno suelto no avisa, y un trabajo bueno pone la cuenta a cero.
        assert_eq!(otra_version(&mut seguidos, CERT_VERSION + 1), None);
        assert_eq!(otra_version(&mut seguidos, CERT_VERSION), None);
        assert_eq!(seguidos, 0);
        for _ in 1..OTRA_VERSION_SEGUIDA {
            assert_eq!(otra_version(&mut seguidos, CERT_VERSION + 1), None);
        }
        let aviso = otra_version(&mut seguidos, CERT_VERSION + 1).unwrap();
        assert!(aviso.contains("certificado") && aviso.contains(&format!("{}", CERT_VERSION + 1)), "{aviso}");
        // Ya avisado: los siguientes no repiten el aviso.
        assert_eq!(otra_version(&mut seguidos, CERT_VERSION + 1), None);
    }
}
