//! Motor de host del minero Pearl de TokenMiner.
//!
//! El contrato con el resto del minero es: `Motor::preparar_trabajo` por trabajo,
//! `Motor::lote_de_vueltas` para minar y `Motor::vaciar` para cambiar de trabajo o parar. Cada candidato
//! lleva su vuelta y su tesela; con la semilla de la vuelta, `share::Constructor` construye la share con el
//! código oficial. `trabajador` une las dos cosas para quien conduzca el motor. La
//! especificación sigue a pearl-research-labs/pearl (licencia ISC).
pub mod cuda;
pub mod motor;
pub mod share;
pub mod trabajador;

pub use motor::{Candidato, Motor, Trabajo, Variante};

pub const K: u32 = 8192;
pub const RANK: u32 = 128;
pub const BM: u32 = 192;
pub const BN: u32 = 192;
pub const WM: u32 = 96;
pub const WN: u32 = 48;
pub const BK: u32 = 128;
/// Cabecera de bloque sin nonce.
pub const HEADER_BYTES: usize = 76;
/// La única versión de certificado que se mina.
pub const CERT_VERSION: u64 = 3;

/// `MiningConfiguration::to_bytes` de la forma 2: k = 8192, rango 128, filas g+{0,8,…,88}, columnas 2t+{0,1,8,9,…,40,41}.
pub const CONFIG_HEX: &str =
    "0020000080000000070b000000000001030500000000000000000000000000000000000000000000000000000000000000000000";

/// h·w · (k/r) · 128 (`penalized_target_bound`): tesela de 12 × 12.
pub const FACTOR: u32 = 12 * 12 * (K / RANK) * 128;

/// BLAKE3(header ‖ configuración).
pub fn job_key(header: &[u8; HEADER_BYTES]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(header);
    h.update(&hex::decode(CONFIG_HEX).expect("CONFIG_HEX es hex"));
    *h.finalize().as_bytes()
}

/// Cota penalizada en little-endian a partir del objetivo del pool (big-endian): objetivo × FACTOR.
/// `None` si no cabe en 256 bits, como el oficial, en vez de saturar.
pub fn cota_penalizada(objetivo_be: &[u8; 32]) -> Option<[u8; 32]> {
    let mut le = *objetivo_be;
    le.reverse();
    let mut out = [0u8; 32];
    let mut arrastre = 0u64;
    for i in 0..32 {
        let v = le[i] as u64 * FACTOR as u64 + arrastre;
        out[i] = v as u8;
        arrastre = v >> 8;
    }
    if arrastre != 0 {
        None
    } else {
        Some(out)
    }
}

/// Palabras little-endian de 32 B.
pub fn palabras(b: &[u8; 32]) -> [u32; 8] {
    std::array::from_fn(|i| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()))
}

/// Semilla de A de la vuelta `r`: la de la vuelta 0 con r sumado a la palabra 0.
pub fn semilla_a_de(semilla_a0: &[u8; 32], r: u32) -> [u8; 32] {
    let mut out = *semilla_a0;
    let w = u32::from_le_bytes(out[..4].try_into().unwrap()).wrapping_add(r);
    out[..4].copy_from_slice(&w.to_le_bytes());
    out
}

#[cfg(test)]
mod pruebas {
    use super::*;

    #[test]
    fn config_de_52_bytes() {
        assert_eq!(hex::decode(CONFIG_HEX).unwrap().len(), 52);
    }

    #[test]
    fn cota_es_objetivo_por_el_factor() {
        let mut t = [0u8; 32];
        t[31] = 1;
        let c = cota_penalizada(&t).unwrap();
        assert_eq!(u32::from_le_bytes(c[..4].try_into().unwrap()), FACTOR);
        assert!(c[4..].iter().all(|&x| x == 0));
        t[31] = 0xFF;
        assert_eq!(
            u32::from_le_bytes(cota_penalizada(&t).unwrap()[..4].try_into().unwrap()),
            255 * FACTOR
        );
    }

    #[test]
    fn cota_que_no_cabe() {
        // FACTOR = 0x120000: el objetivo debe quedar por debajo de 2²⁵⁶ / FACTOR.
        let mut t = [0u8; 32];
        t[0] = 0x20;
        assert_eq!(cota_penalizada(&t), None);
        let mut t = [0u8; 32];
        t[2] = 0x20;
        assert_eq!(cota_penalizada(&t), None);
        let mut t = [0u8; 32];
        t[2] = 0x0E;
        assert_eq!(cota_penalizada(&t).unwrap()[31], 0xFC);
    }

    #[test]
    fn semilla_de_vuelta_da_la_vuelta() {
        let mut s = [0u8; 32];
        s[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        s[31] = 9;
        let r = semilla_a_de(&s, 2);
        assert_eq!(u32::from_le_bytes(r[..4].try_into().unwrap()), 1);
        assert_eq!(r[31], 9);
    }
}
