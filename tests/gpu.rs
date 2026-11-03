//! Vuelta completa en GPU contra el código oficial (`pearl-verifier round`). Necesita la GPU libre:
//! cargo test --offline --release -- --ignored --test-threads=1
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use tokenminer_pearl::{
    cota_penalizada, job_key, semilla_a_de, Motor, Trabajo, Variante, K, RANK, WM, WN,
};

fn verificador() -> PathBuf {
    // PEARL_VERIFIER, o el de tools/verifier compilado con
    // `cargo build --release --manifest-path tools/verifier/Cargo.toml`.
    std::env::var_os("PEARL_VERIFIER")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tools/verifier/target/release/pearl-verifier")
                .with_extension(std::env::consts::EXE_EXTENSION)
        })
}

const CABECERA: [u8; 76] = [0x5C; 76];
const SEMILLA_A0: [u8; 32] = [0xA7; 32];
const SEMILLA_B: [u8; 32] = [0x3D; 32];

fn referencia(m: u32, n: u32, semilla_a: &[u8; 32]) -> (Value, HashMap<(u32, u32), String>) {
    let dir = std::env::temp_dir().join("pl02-gpu-ref");
    let salida = Command::new(verificador())
        .args([
            "round",
            &hex::encode(CABECERA),
            &m.to_string(),
            &n.to_string(),
            &K.to_string(),
            &RANK.to_string(),
            &hex::encode(semilla_a),
            &hex::encode(SEMILLA_B),
            dir.to_str().unwrap(),
            "2",
        ])
        .output()
        .expect("no se pudo lanzar pearl-verifier");
    assert!(
        salida.status.success(),
        "{}",
        String::from_utf8_lossy(&salida.stderr)
    );
    let r: Value = serde_json::from_slice(&salida.stdout).unwrap();
    let teselas = r["tiles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            (
                (t[0].as_u64().unwrap() as u32, t[1].as_u64().unwrap() as u32),
                t[2].as_str().unwrap().to_string(),
            )
        })
        .collect();
    (r, teselas)
}

/// Con la cota máxima todas las teselas son candidatas: cada hash debe ser el oficial.
fn comprobar(m: u32, n: u32, vueltas: u32, variante: Option<Variante>) {
    let mut motor = Motor::con_variante(m, n, 1 << 16, variante).unwrap();
    let clave = job_key(&CABECERA);
    motor
        .preparar_trabajo(&Trabajo {
            job_key: clave,
            semilla_b: SEMILLA_B,
            semilla_a0: SEMILLA_A0,
            cota: [0xFF; 32],
        })
        .unwrap();
    let total = (motor.mu / WM * 8) * (motor.nu / WN * 4);
    for v in 1..=vueltas {
        motor.lote_de_vueltas(1).unwrap();
        let candidatos = motor.vaciar().unwrap();
        let (r, teselas) = referencia(m, n, &semilla_a_de(&SEMILLA_A0, v));
        assert_eq!(r["jobKey"].as_str().unwrap(), hex::encode(clave));
        assert_eq!(
            candidatos.len() as u32,
            total,
            "vuelta {v}: todas las teselas son candidatas"
        );
        for c in &candidatos {
            assert_eq!(c.vuelta, v);
            assert_eq!(
                teselas.get(&(c.fila, c.columna)).map(String::as_str),
                Some(hex::encode(c.hash).as_str()),
                "vuelta {v}, fila {}, columna {}",
                c.fila,
                c.columna
            );
        }
    }
}

#[test]
#[ignore = "necesita la GPU"]
fn vuelta_contra_el_oficial() {
    comprobar(256, 512, 3, None);
}

#[test]
#[ignore = "necesita la GPU"]
fn variante_de_las_demas_rtx_contra_el_oficial() {
    comprobar(256, 512, 2, Some(Variante::CpAsync));
    tokenminer_pearl::trabajador::autocomprobar(Some(Variante::CpAsync)).unwrap();
}

#[test]
#[ignore = "necesita la GPU"]
fn matriz_grande_contra_el_oficial() {
    // Más de 1024·128 porciones en Bᵀ: pasa por pl02_parents.
    comprobar(256, 32768, 1, None);
}

#[test]
#[ignore = "necesita la GPU"]
fn la_autocomprobacion_pasa_y_detecta_una_tesela_cambiada() {
    tokenminer_pearl::trabajador::autocomprobar(None).unwrap();
    let mut motor = Motor::nuevo(256, 512, 1 << 16).unwrap();
    motor
        .preparar_trabajo(&Trabajo {
            job_key: job_key(&CABECERA),
            semilla_b: SEMILLA_B,
            semilla_a0: SEMILLA_A0,
            cota: [0xFF; 32],
        })
        .unwrap();
    motor.lote_de_vueltas(1).unwrap();
    let mut c = motor.vaciar().unwrap();
    c.sort_by_key(|x| {
        let mut h = x.hash;
        h.reverse();
        h
    });
    // La mejor tesela con la prueba de la peor no pasa.
    let (mejor, peor) = (&c[0], c.last().unwrap());
    let prueba = tokenminer_pearl::share::Constructor::default()
        .prueba(
            &CABECERA,
            256,
            512,
            &semilla_a_de(&SEMILLA_A0, peor.vuelta),
            peor.fila,
            peor.columna,
        )
        .unwrap();
    assert!(tokenminer_pearl::share::verifica_justo(&CABECERA, &prueba, &mejor.hash).is_err());
    assert!(tokenminer_pearl::share::verifica_justo(&CABECERA, &prueba, &peor.hash).is_ok());
}

#[test]
#[ignore = "necesita la GPU"]
fn cada_gpu_tiene_su_luid_de_windows() {
    let gpus = tokenminer_pearl::cuda::gpus().unwrap();
    println!("{gpus:?}");
    assert!(!gpus.is_empty());
    for luid in &gpus {
        let (alto, bajo) = luid.split_once('_').unwrap();
        assert!(
            alto.starts_with("0x")
                && bajo.starts_with("0x")
                && alto.len() == 10
                && bajo.len() == 10,
            "{luid}"
        );
    }
}

/// Con límite, las vueltas por segundo bajan en proporción. `PEARL_LIMITES=100,50,…` y `PEARL_SEGUNDOS` cambian la
/// tanda, para mirar a la vez la ocupación que ve Windows.
#[test]
#[ignore = "necesita la GPU"]
fn el_limite_reparte_el_tiempo_de_gpu() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{mpsc, Arc};
    use std::time::{Duration, Instant};
    use tokenminer_pearl::trabajador::{trabajar, Contadores, Encargo, Orden};
    use tokenminer_pearl::CERT_VERSION;
    let limites: Vec<u32> = std::env::var("PEARL_LIMITES")
        .unwrap_or("100,50,20".into())
        .split(',')
        .map(|l| l.trim().parse().unwrap())
        .collect();
    let segundos: u64 = std::env::var("PEARL_SEGUNDOS")
        .map(|s| s.parse().unwrap())
        .unwrap_or(6);
    let (vigente, c) = (Arc::new(AtomicU64::new(0)), Arc::new(Contadores::default()));
    let (ordenes, ord_rx) = mpsc::channel();
    let (av_tx, av_rx) = mpsc::channel();
    let motor = {
        let (vigente, c) = (vigente.clone(), c.clone());
        std::thread::spawn(move || trabajar(0, 2048, 0, 8, None, ord_rx, vigente, c, av_tx))
    };
    vigente.store(1, Ordering::SeqCst);
    ordenes
        .send(Orden::Trabajo(Encargo {
            revision: 1,
            job_id: "prueba".into(),
            cabecera: CABECERA,
            objetivo: [0; 32],
            cert_version: CERT_VERSION,
        }))
        .unwrap();
    while c.rounds.load(Ordering::Relaxed) == 0 {
        std::thread::sleep(Duration::from_millis(50));
    }
    let mut tasas = Vec::new();
    for &limite in &limites {
        c.limite.store(limite, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(500));
        let (r0, t0) = (c.rounds.load(Ordering::Relaxed), Instant::now());
        std::thread::sleep(Duration::from_secs(segundos));
        let tasa = (c.rounds.load(Ordering::Relaxed) - r0) as f64 / t0.elapsed().as_secs_f64();
        println!("límite {limite:>3} %: {tasa:.1} vueltas/s");
        tasas.push(tasa);
    }
    c.apartado.store(true, Ordering::Relaxed);
    std::thread::sleep(Duration::from_millis(500));
    let r0 = c.rounds.load(Ordering::Relaxed);
    std::thread::sleep(Duration::from_secs(2));
    assert_eq!(
        c.rounds.load(Ordering::Relaxed),
        r0,
        "apartado no debe lanzar vueltas"
    );
    ordenes.send(Orden::Parar).unwrap();
    motor.join().unwrap().unwrap();
    drop(av_rx);
    let lleno = tasas[limites
        .iter()
        .position(|&l| l == 100)
        .expect("falta el 100 %")];
    for (&limite, &tasa) in limites.iter().zip(&tasas).filter(|(&l, _)| l < 100) {
        let parte = tasa / lleno * 100.0;
        assert!(
            parte < limite as f64 + 5.0 && parte > limite as f64 * 0.8,
            "límite {limite} %: {parte:.0} % de la tasa"
        );
    }
}

#[test]
#[ignore = "necesita la GPU"]
fn memoria_escasa_elige_n_pequena() {
    let mut motor = Motor::nuevo(2048, 4096, 16).unwrap();
    let objetivo = [0u8; 32];
    let cota = cota_penalizada(&objetivo).unwrap();
    motor
        .preparar_trabajo(&Trabajo {
            job_key: [1; 32],
            semilla_b: SEMILLA_B,
            semilla_a0: SEMILLA_A0,
            cota,
        })
        .unwrap();
    motor.lote_de_vueltas(4).unwrap();
    assert!(motor.vaciar().unwrap().is_empty());
}
