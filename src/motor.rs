//! Motor de minería en GPU, certificado contra el código oficial de Pearl.
//!
//! Por trabajo: B, su raíz, b_seed, ruido y B'ᵀ. Por vuelta: A desde la semilla de la vuelta, raíz, salado,
//! a_seed, ruido, A' y el producto con epílogo. Todo va en un stream; el host sólo espera, sin girar, a que
//! acabe un lote para leer sus candidatos mientras la GPU hace el siguiente.
//!
//! Es la forma 2: A y Bᵀ valen 64 salvo la última porción de A, que lleva la receta de la vuelta, A' y B'
//! van por teselas de carga, y el producto usa sólo los bloques enteros (`mu` filas y `nu` columnas de las
//! m y n declaradas).
//!
//! Hay dos variantes de los kernels. `Tma`, la rápida, es cubin de sm_120. `CpAsync` es PTX de sm_80 que compila
//! el driver, para las demás RTX: sin TMA, con cp.async y A' y B' por filas. En una RTX 4060 portátil da unos 46 TMAC/s
//! (el techo de `mma.sync` medido en esa tarjeta es de unos 54); en sm_120 va mucho más lento que `Tma` (28,7 TMAC/s).
//! Las dos pasan la autocomprobación antes de minar. El pliegue XOR de cada tramo se compila con `PL_FOLD_ROWS=1`: con el
//! árbol sobre un vector, el compilador lo deja en memoria local y el pliegue tarda tanto como el producto.
use crate::cuda::{Arg, Cuda, Ptr, TmaMap};
use crate::{BK, BM, BN, K, RANK, WM, WN};

const CORE: &[u8] = include_bytes!("../kernels/pl02_core.sm_120.cubin");
const ROUND: &[u8] = include_bytes!("../kernels/pl02_round.sm_120.cubin");
const CORE_CP: &[u8] = include_bytes!("../kernels/pl02_core_cp.compute_80.ptx");
const ROUND_CP: &[u8] = include_bytes!("../kernels/pl02_round.compute_80.ptx");
const STAGES: u32 = 2;
const NTHREADS: u32 = (BM / WM) * (BN / WN) * 32;
const CAND_WORDS: usize = 12; // Candidate: round_id, row0, col0, flags, hash[8]

/// Cómo llegan A' y B' a la memoria compartida.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variante {
    Tma,
    CpAsync,
}

impl Variante {
    fn shared(self) -> u32 {
        let etapas = STAGES * (BM + BN) * BK;
        if self == Variante::Tma {
            etapas + STAGES * 8 * 2
        } else {
            etapas
        }
    }

    /// La que corresponde a una arquitectura: TMA en sm_120 y cp.async desde sm_80.
    pub fn para(arquitectura: (i32, i32)) -> Option<Variante> {
        match arquitectura {
            (12, 0) => Some(Variante::Tma),
            (mayor, _) if mayor >= 8 => Some(Variante::CpAsync),
            _ => None,
        }
    }
}

/// Una tesela que cumple la cota. Con (vuelta, fila, columna) y las semillas del trabajo se reconstruye la share.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidato {
    pub vuelta: u32,
    pub fila: u32,
    pub columna: u32,
    pub hash: [u8; 32],
}

/// Lo que define un trabajo: semilla de Bᵀ, semilla de A de la vuelta 0 y cota penalizada (little-endian).
#[derive(Clone, Debug)]
pub struct Trabajo {
    pub job_key: [u8; 32],
    pub semilla_b: [u8; 32],
    pub semilla_a0: [u8; 32],
    pub cota: [u8; 32],
}

struct Kernels {
    ronda: Ptr,
    chunks: Ptr,
    parents: Ptr,
    tree_final: Ptr,
    tables: Ptr,
    noised: Ptr,
    reset: Ptr,
    core: Ptr,
}

pub struct Motor {
    cu: Cuda,
    k: Kernels,
    pub variante: Variante,
    /// Multiprocesadores de la GPU, para el techo teórico (`techo_macs`).
    pub sm: u32,
    pub m: u32,
    pub n: u32,
    pub mu: u32,
    pub nu: u32,
    max_cands: u32,
    job: u64,
    rd: u64,
    mp: u64,
    cv0: u64,
    cv1: u64,
    root_a: u64,
    root_b: u64,
    espina: u64,
    ua: u64,
    ub: u64,
    pa: u64,
    pb: u64,
    ap: u64,
    btp: u64,
    cands: [u64; 2],
    nc: [u64; 2],
    mapas: Option<[TmaMap; 2]>,
    s: Ptr,
    ev: [Ptr; 2],
    lote: u32,
}

/// Columnas de B para esta memoria libre: la mayor potencia de dos ≤ 131.072 que deja la mitad libre.
/// Vueltas más largas reparten su preparación entre más multiplicaciones.
pub fn n_para_memoria(libre: usize) -> u32 {
    let mut n = 131072u32;
    // B'ᵀ (n·k) y su tabla uniforme dominan; A', árbol y candidatos son ~20 MB.
    while n > 4096 && n as usize * (K as usize + 128) + (64 << 20) > libre / 2 {
        n /= 2;
    }
    n
}

impl Motor {
    /// `n` = 0 lo elige según la memoria libre de la GPU. La variante sale de la arquitectura de la GPU.
    pub fn nuevo(m: u32, n: u32, max_cands: u32) -> Result<Motor, String> {
        Self::con_variante(m, n, max_cands, None)
    }

    /// Como `nuevo`, con la variante forzada para probarla. La de TMA sólo carga en sm_120.
    pub fn con_variante(
        m: u32,
        n: u32,
        max_cands: u32,
        variante: Option<Variante>,
    ) -> Result<Motor, String> {
        Self::en(0, m, n, max_cands, variante)
    }

    /// Como `con_variante`, en la GPU `dispositivo`.
    pub fn en(
        dispositivo: i32,
        m: u32,
        n: u32,
        max_cands: u32,
        variante: Option<Variante>,
    ) -> Result<Motor, String> {
        let mut cu = Cuda::en(dispositivo)?;
        let variante = match variante {
            Some(v) => v,
            None => Variante::para(cu.arquitectura()?)
                .ok_or("Pearl necesita una NVIDIA RTX 30 o posterior")?,
        };
        if variante.shared() > cu.compartida_maxima()? {
            return Err("La GPU no tiene memoria compartida suficiente para Pearl".into());
        }
        let n = if n == 0 {
            n_para_memoria(cu.memoria()?.0)
        } else {
            n
        };
        for (d, nombre) in [(m, "m"), (n, "n")] {
            let porciones = d as u64 * K as u64 / 1024;
            if !porciones.is_power_of_two() || porciones < 256 {
                return Err(format!(
                    "{nombre}·k/1024 debe ser potencia de dos y ≥ 256 (árbol BLAKE3 perfecto)"
                ));
            }
        }
        let (mu32, nu32) = (m / BM * BM, n / BN * BN);
        if mu32 == m {
            return Err(
                "la última fila de A lleva la receta: m no puede ser múltiplo de BM".into(),
            );
        }
        let tma = variante == Variante::Tma;
        let r = cu.cargar(if tma { ROUND } else { ROUND_CP })?;
        let c = cu.cargar(if tma { CORE } else { CORE_CP })?;
        let k = Kernels {
            ronda: cu.funcion(r, "pl02_ronda", 0)?,
            chunks: cu.funcion(r, "pl02_chunks", 0)?,
            parents: cu.funcion(r, "pl02_parents", 0)?,
            tree_final: cu.funcion(r, "pl02_tree_final", 0)?,
            tables: cu.funcion(r, "pl02_tables", 0)?,
            noised: cu.funcion(r, "pl02_noised", 0)?,
            reset: cu.funcion(r, "pl02_reset", 0)?,
            core: cu.funcion(c, "pl02_mine", variante.shared())?,
        };
        let (ku, mu, nu) = (K as usize, mu32 as usize, nu32 as usize);
        let job = cu.malloc(48 * 4)?;
        let (rd, mp) = (cu.malloc(64)?, cu.malloc(96)?);
        let max_cv = m.max(n) as usize * ku / 1024 / 128;
        let (cv0, cv1) = (cu.malloc(max_cv * 32)?, cu.malloc(max_cv * 32)?);
        let (root_a, root_b) = (cu.malloc(32)?, cu.malloc(32)?);
        let espina = cu.malloc(32 * 32)?;
        let (ua, ub) = (
            cu.malloc(mu * RANK as usize)?,
            cu.malloc(nu * RANK as usize)?,
        );
        let (pa, pb) = (cu.malloc(ku * 2)?, cu.malloc(ku * 2)?);
        let (ap, btp) = (cu.malloc(mu * ku)?, cu.malloc(nu * ku)?);
        let bytes_cand = max_cands as usize * CAND_WORDS * 4;
        let cands = [cu.malloc(bytes_cand)?, cu.malloc(bytes_cand)?];
        let nc = [cu.malloc(4)?, cu.malloc(4)?];
        // Por teselas de carga: tensores de filas de BK bytes (PL_TILED).
        let (fm, fn_) = ((mu * ku) as u64 / BK as u64, (nu * ku) as u64 / BK as u64);
        let mapas = if tma {
            Some([
                cu.tensor_map(ap, BK as u64, fm, BK, BM)?,
                cu.tensor_map(btp, BK as u64, fn_, BK, BN)?,
            ])
        } else {
            None
        };
        let s = cu.stream()?;
        let ev = [cu.evento()?, cu.evento()?];
        let sm = cu.multiprocesadores()?;
        Ok(Motor {
            cu,
            k,
            variante,
            sm,
            m,
            n,
            mu: mu32,
            nu: nu32,
            max_cands,
            job,
            rd,
            mp,
            cv0,
            cv1,
            root_a,
            root_b,
            espina,
            ua,
            ub,
            pa,
            pb,
            ap,
            btp,
            cands,
            nc,
            mapas,
            s,
            ev,
            lote: 0,
        })
    }

    fn lanzar(&self, f: Ptr, grid: u32, block: u32, args: &[Arg]) -> Result<(), String> {
        self.cu.lanzar(f, grid, block, args, 0, self.s)
    }

    /// Raíz de la matriz, salado y semilla de ruido (ver pl02_round.cu).
    fn merkle(
        &self,
        semilla: u64,
        filas: u32,
        lado: i32,
        raiz: u64,
        espina: u64,
    ) -> Result<(), String> {
        let total = (filas as u64 * K as u64 / 1024) as u32;
        let mut cnt = total / 128;
        let ultima = if lado == 0 { total - 1 } else { u32::MAX }; // sólo A lleva receta, en su última porción
        self.lanzar(
            self.k.chunks,
            cnt,
            128,
            &[
                Arg::Mem(semilla),
                Arg::Mem(self.job),
                Arg::Mem(self.cv0),
                Arg::U32(ultima),
                Arg::Mem(espina),
            ],
        )?;
        let (mut src, mut dst) = (self.cv0, self.cv1);
        while cnt > 1024 {
            cnt /= 2;
            self.lanzar(
                self.k.parents,
                cnt.div_ceil(128),
                128,
                &[
                    Arg::Mem(src),
                    Arg::Mem(dst),
                    Arg::U32(cnt),
                    Arg::Mem(self.job),
                ],
            )?;
            std::mem::swap(&mut src, &mut dst);
        }
        self.lanzar(
            self.k.tree_final,
            1,
            cnt / 2,
            &[
                Arg::Mem(src),
                Arg::U32(cnt),
                Arg::Mem(self.job),
                Arg::Mem(self.rd),
                Arg::I32(lado),
                Arg::Mem(self.mp),
                Arg::Mem(raiz),
                Arg::Mem(espina),
            ],
        )
    }

    fn ruido(
        &self,
        clave: u64,
        lado: i32,
        filas: u32,
        u: u64,
        parejas: u64,
        semilla: u64,
        dst: u64,
    ) -> Result<(), String> {
        let total = filas * RANK / 32 + K / 8;
        self.lanzar(
            self.k.tables,
            total.div_ceil(128),
            128,
            &[
                Arg::Mem(clave),
                Arg::I32(lado),
                Arg::U32(filas),
                Arg::Mem(u),
                Arg::Mem(parejas),
            ],
        )?;
        // El último parámetro es tile_rows, las filas de un bloque; 0 = por filas.
        let teselas = if self.variante != Variante::Tma {
            0
        } else if lado == 1 {
            BN
        } else {
            BM
        };
        self.lanzar(
            self.k.noised,
            filas,
            K / 16,
            &[
                Arg::Mem(semilla),
                Arg::Mem(u),
                Arg::Mem(parejas),
                Arg::Mem(dst),
                Arg::U32(teselas),
            ],
        )
    }

    /// Nuevo trabajo: B fija y contador de vueltas a 0. Espera a que acabe lo anterior.
    pub fn preparar_trabajo(&mut self, t: &Trabajo) -> Result<(), String> {
        self.cu.sincronizar()?;
        self.lote = 0;
        let mut job = [0u32; 48];
        for (i, parte) in [&t.job_key, &t.semilla_b, &t.semilla_a0, &t.cota]
            .iter()
            .enumerate()
        {
            job[i * 8..(i + 1) * 8].copy_from_slice(&crate::palabras(parte));
        }
        job[40] = self.m;
        job[41] = self.n;
        self.cu.htod(self.job, &job)?;
        self.cu.a_cero(self.rd, 64)?;
        self.merkle(self.job + 32, self.n, 1, self.root_b, 0)?;
        self.ruido(
            self.job + 128,
            1,
            self.nu,
            self.ub,
            self.pb,
            self.job + 32,
            self.btp,
        )?;
        // Los hermanos del camino de la última porción de A: fijos mientras dure el trabajo.
        self.merkle(self.rd, self.m, 0, self.root_a, self.espina)?;
        self.cu.registrar(self.ev[0], self.s)?;
        self.cu.esperar(self.ev[0])
    }

    fn vuelta(&self, b: usize) -> Result<(), String> {
        let porciones = self.m * (K / 1024);
        self.lanzar(
            self.k.ronda,
            1,
            1,
            &[
                Arg::Mem(self.job),
                Arg::Mem(self.rd),
                Arg::Mem(self.espina),
                Arg::U32(porciones),
                Arg::U32(porciones.trailing_zeros()),
                Arg::Mem(self.mp),
                Arg::Mem(self.root_a),
            ],
        )?;
        self.ruido(self.mp, 0, self.mu, self.ua, self.pa, self.rd, self.ap)?;
        let mut args = vec![
            Arg::Mem(self.ap),
            Arg::Mem(self.btp),
            Arg::I32(self.mu as i32),
            Arg::I32(self.nu as i32),
            Arg::Mem(self.mp),
            Arg::Mem(0),
            Arg::Mem(self.cands[b]),
            Arg::Mem(self.nc[b]),
            Arg::U32(self.max_cands),
        ];
        if let Some([a, bt]) = self.mapas {
            args.extend([Arg::Map(a), Arg::Map(bt)]);
        }
        self.cu.lanzar(
            self.k.core,
            (self.mu / BM) * (self.nu / BN),
            NTHREADS,
            &args,
            self.variante.shared(),
            self.s,
        )
    }

    /// Encola `vueltas` vueltas y devuelve los candidatos del lote anterior; la GPU no espera al host.
    pub fn lote_de_vueltas(&mut self, vueltas: u32) -> Result<Vec<Candidato>, String> {
        let b = (self.lote % 2) as usize;
        self.lanzar(self.k.reset, 1, 1, &[Arg::Mem(self.nc[b])])?;
        for _ in 0..vueltas {
            self.vuelta(b)?;
        }
        self.cu.registrar(self.ev[b], self.s)?;
        self.lote += 1;
        if self.lote == 1 {
            Ok(vec![])
        } else {
            self.leer(1 - b)
        }
    }

    /// Espera al último lote y devuelve sus candidatos (para cambiar de trabajo o parar).
    pub fn vaciar(&mut self) -> Result<Vec<Candidato>, String> {
        if self.lote == 0 {
            return Ok(vec![]);
        }
        let b = ((self.lote - 1) % 2) as usize;
        self.lote = 0;
        self.leer(b)
    }

    fn leer(&self, b: usize) -> Result<Vec<Candidato>, String> {
        self.cu.esperar(self.ev[b])?;
        let mut n = [0u32; 1];
        self.cu.dtoh(&mut n, self.nc[b])?;
        let k = n[0].min(self.max_cands) as usize;
        if k == 0 {
            return Ok(vec![]);
        }
        let mut buf = vec![0u32; k * CAND_WORDS];
        self.cu.dtoh(&mut buf, self.cands[b])?;
        Ok(buf.chunks_exact(CAND_WORDS).map(candidato).collect())
    }
}

fn candidato(c: &[u32]) -> Candidato {
    let mut hash = [0u8; 32];
    for (i, w) in c[4..12].iter().enumerate() {
        hash[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    Candidato {
        vuelta: c[0],
        fila: c[1],
        columna: c[2],
        hash,
    }
}

#[cfg(test)]
mod pruebas {
    use super::*;

    #[test]
    fn candidato_little_endian() {
        let c = candidato(&[7, 8, 16, 0, 0x04030201, 0, 0, 0, 0, 0, 0, 0x88776655]);
        assert_eq!((c.vuelta, c.fila, c.columna), (7, 8, 16));
        assert_eq!(&c.hash[..4], &[1, 2, 3, 4]);
        assert_eq!(&c.hash[28..], &[0x55, 0x66, 0x77, 0x88]);
    }

    #[test]
    fn variante_por_arquitectura() {
        assert_eq!(Variante::para((12, 0)), Some(Variante::Tma));
        assert_eq!(Variante::para((8, 9)), Some(Variante::CpAsync));
        assert_eq!(Variante::para((8, 6)), Some(Variante::CpAsync));
        assert_eq!(Variante::para((7, 5)), None);
    }

    #[test]
    fn memoria_elige_n() {
        assert_eq!(n_para_memoria(7013 << 20), 131072);
        assert_eq!(n_para_memoria(1300 << 20), 65536);
        assert_eq!(n_para_memoria(500 << 20), 16384);
        assert_eq!(n_para_memoria(200 << 20), 4096);
        assert_eq!(n_para_memoria(0), 4096);
    }
}
