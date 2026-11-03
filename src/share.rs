//! Construcción de shares con el código oficial de Pearl (`zk-pow` y `pearl-blake3`, ISC).
//!
//! Forma 2: A y Bᵀ valen 64 salvo la última porción de A, que lleva la receta de la vuelta. Con la cabecera, m, n,
//! la semilla de A de la vuelta y la tesela se reconstruye todo. El árbol de Bᵀ se guarda mientras dure el trabajo y
//! el de A mientras no cambie la vuelta. Bᵀ llega a 1 GB, pero al ser constante basta con sus hashes (`ArbolCte`).
use crate::{palabras, BM, BN, HEADER_BYTES, K, RANK, WM, WN};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use blake3::hazmat::{merge_subtrees_non_root, merge_subtrees_root, HasherExt, Mode};
use pearl_blake3::{MerkleProof, MerkleTree};
use primitive_types::U256;
use std::collections::BTreeSet;
use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern};
use zk_pow::api::verify::verify_plain_proof;
use zk_pow::ffi::plain_proof::{check_cert_version_eligible, MatrixMerkleProof, PlainProof};

const H: usize = (WM / 8) as usize;
const W: usize = (2 * WN / 8) as usize;
const PORCION: usize = 1024;
const BYTE_CTE: u8 = 64;

/// `MiningConfiguration` de la forma 2.
pub fn configuracion() -> Result<MiningConfiguration, String> {
    let filas: Vec<u32> = (0..H as u32).map(|i| i * 8).collect();
    let columnas: Vec<u32> = (0..W as u32 / 2).flat_map(|i| [i * 8, i * 8 + 1]).collect();
    let patron = |lista: &[u32]| PeriodicPattern::from_list(lista).map_err(|e| e.to_string());
    Ok(MiningConfiguration {
        common_dim: K,
        rank: RANK as u16,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: patron(&filas)?,
        cols_pattern: patron(&columnas)?,
        moe: None,
    })
}

/// Escribe en `out` la receta desde la palabra `desde`, como `gen_word` de pl02_round.cu.
fn receta(seed: &[u32; 8], out: &mut [u8], desde: usize) {
    for (w, b) in out.chunks_exact_mut(4).enumerate() {
        let i = desde + w;
        let mut h = (i as u32).wrapping_add(seed[i % 8]);
        h = (h ^ (h >> 16)).wrapping_mul(0x7feb352d);
        h = (h ^ (h >> 15)).wrapping_mul(0x846ca68b);
        h ^= h >> 16;
        for (j, x) in b.iter_mut().enumerate() {
            *x = (((h >> (j * 8)) & 127) as i32 - 64) as i8 as u8;
        }
    }
}

/// A de la vuelta: 64 en todos los bytes salvo la última porción de 1024 B.
fn matriz_a(semilla: &[u8; 32], m: usize) -> Vec<u8> {
    let mut a = vec![64u8; m * K as usize];
    let desde = a.len() - 1024;
    receta(&palabras(semilla), &mut a[desde..], desde / 4);
    a
}

/// Árbol Merkle de una matriz con todos los bytes a 64: sólo los hashes, sin los datos. Da lo mismo que
/// `pearl_blake3::MerkleTree` (la prueba `arbol_cte_igual_al_oficial` lo contrasta), con 2·32 B por porción.
/// El número de porciones es potencia de dos.
struct ArbolCte {
    capas: Vec<Vec<[u8; 32]>>,
    raiz: [u8; 32],
}

impl ArbolCte {
    fn nuevo(clave: &[u8; 32], porciones: usize) -> ArbolCte {
        assert!(porciones.is_power_of_two() && porciones >= 2);
        let mut hojas = vec![[0u8; 32]; porciones];
        let hilos = std::thread::available_parallelism().map_or(1, |n| n.get());
        std::thread::scope(|s| {
            for (t, trozo) in hojas.chunks_mut(porciones.div_ceil(hilos)).enumerate() {
                let base = t * porciones.div_ceil(hilos);
                s.spawn(move || {
                    let dato = [BYTE_CTE; PORCION];
                    for (i, h) in trozo.iter_mut().enumerate() {
                        let mut hasher = blake3::Hasher::new_keyed(clave);
                        hasher.set_input_offset(((base + i) * PORCION) as u64);
                        hasher.update(&dato);
                        *h = hasher.finalize_non_root();
                    }
                });
            }
        });
        let modo = Mode::KeyedHash(clave);
        let mut capas = vec![hojas];
        while capas.last().unwrap().len() > 2 {
            let sube = capas
                .last()
                .unwrap()
                .chunks_exact(2)
                .map(|p| merge_subtrees_non_root(&p[0], &p[1], modo))
                .collect();
            capas.push(sube);
        }
        let ultima = capas.last().unwrap();
        let raiz = *merge_subtrees_root(&ultima[0], &ultima[1], modo).as_bytes();
        ArbolCte { capas, raiz }
    }

    /// Como `MerkleTree::get_multileaf_proof`: mismos hermanos, en el mismo orden.
    fn prueba(&self, indices: &[usize]) -> MerkleProof {
        let unicos: BTreeSet<usize> = indices.iter().copied().collect();
        let total_leaves = self.capas[0].len();
        let leaf_indices: Vec<usize> = unicos.iter().copied().collect();
        let mut siblings = Vec::new();
        let mut actuales = unicos;
        for capa in &self.capas {
            for &i in &actuales {
                let hermano = i ^ 1;
                if !actuales.contains(&hermano) {
                    siblings.push(capa[hermano]);
                }
            }
            actuales = actuales.iter().map(|&i| i / 2).collect();
        }
        MerkleProof {
            leaf_data: vec![[BYTE_CTE; PORCION]; leaf_indices.len()],
            leaf_indices,
            total_leaves,
            root: self.raiz,
            siblings,
        }
    }
}

fn parte(arbol: &MerkleTree, filas: Vec<usize>, dim: (usize, usize)) -> MatrixMerkleProof {
    let hojas = MerkleTree::compute_leaf_indices_from_rows(&filas, dim);
    MatrixMerkleProof {
        proof: arbol.get_multileaf_proof(&hojas),
        row_indices: filas,
    }
}

#[derive(Default)]
pub struct Constructor {
    arbol_b: Option<([u8; 32], usize, ArbolCte)>,
    arbol_a: Option<([u8; 32], [u8; 32], MerkleTree)>,
}

impl Constructor {
    /// Olvida los árboles: al parar no hay por qué retener la memoria.
    pub fn soltar(&mut self) {
        self.arbol_a = None;
        self.arbol_b = None;
    }

    /// `PlainProof` serializado en base64 de la tesela con fila base `fila` y columna base `columna`.
    pub fn prueba(
        &mut self,
        cabecera: &[u8; HEADER_BYTES],
        m: u32,
        n: u32,
        semilla_a: &[u8; 32],
        fila: u32,
        columna: u32,
    ) -> Result<String, String> {
        let (mu, nu) = ((m / BM * BM) as usize, (n / BN * BN) as usize);
        let (m, n, k, fila, columna) = (
            m as usize,
            n as usize,
            K as usize,
            fila as usize,
            columna as usize,
        );
        if fila % (8 * H) >= 8 || fila + 8 * (H - 1) >= mu {
            return Err("fila base fuera de patrón".into());
        }
        if columna % (4 * W) >= 8 || columna % 2 != 0 || columna + 8 * (W / 2 - 1) + 1 >= nu {
            return Err("columna base fuera de patrón".into());
        }
        let clave = crate::job_key(cabecera);
        if !matches!(&self.arbol_b, Some((c, x, _)) if *c == clave && *x == n) {
            self.arbol_b = Some((clave, n, ArbolCte::nuevo(&clave, n * k / PORCION)));
        }
        if !matches!(&self.arbol_a, Some((c, s, _)) if *c == clave && s == semilla_a) {
            self.arbol_a = Some((
                clave,
                *semilla_a,
                MerkleTree::new(&matriz_a(semilla_a, m), clave),
            ));
        }
        let filas: Vec<usize> = (0..H).map(|i| fila + 8 * i).collect();
        let columnas: Vec<usize> = (0..W / 2)
            .flat_map(|i| [columna + 8 * i, columna + 8 * i + 1])
            .collect();
        let hojas_b = MerkleTree::compute_leaf_indices_from_rows(&columnas, (n, k));
        let bt = MatrixMerkleProof {
            proof: self.arbol_b.as_ref().unwrap().2.prueba(&hojas_b),
            row_indices: columnas,
        };
        let prueba = PlainProof {
            m,
            n,
            k,
            noise_rank: RANK as usize,
            a: parte(&self.arbol_a.as_ref().unwrap().2, filas, (m, k)),
            bt,
            moe: None,
        };
        Ok(STANDARD.encode(bincode::serialize(&prueba).map_err(|e| e.to_string())?))
    }
}

/// `nbits` compacto más pequeño cuya cota penalizada alcanza `hash` (little-endian).
fn nbits_para(hash: &[u8; 32]) -> u32 {
    let objetivo = U256::from_little_endian(hash) / crate::FACTOR + 1;
    let mut bytes = objetivo.bits().div_ceil(8) as u32;
    let mut mantisa = if bytes <= 3 {
        objetivo.low_u32() << (8 * (3 - bytes))
    } else {
        ((objetivo >> (8 * (bytes - 3))).low_u32()) + 1
    };
    if mantisa >= 0x80_0000 {
        mantisa = (mantisa >> 8) + 1;
        bytes += 1;
    }
    (bytes << 24) | mantisa
}

/// La share pasa el verificador oficial con la dificultad justa para `hash`. Una tesela mal calculada daría otro
/// hash, y lo normal es que no llegara a esa cota.
pub fn verifica_justo(
    cabecera: &[u8; HEADER_BYTES],
    prueba_b64: &str,
    hash: &[u8; 32],
) -> Result<(), String> {
    let cab = IncompleteBlockHeader::from_bytes(cabecera).map_err(|e| e.to_string())?;
    let bytes = STANDARD.decode(prueba_b64).map_err(|e| e.to_string())?;
    let prueba = PlainProof::deserialize_compat(&bytes).map_err(|e| e.to_string())?;
    let version = check_cert_version_eligible(crate::CERT_VERSION as u32, &prueba)
        .map_err(|e| e.to_string())?;
    verify_plain_proof(
        &cab,
        &prueba,
        Some(nbits_para(hash)),
        version.seed_derivation(),
    )
    .map_err(|e| format!("{e:#}"))
}

/// Comprueba la cabecera con el formato oficial.
pub fn cabecera_valida(cabecera: &[u8; HEADER_BYTES]) -> bool {
    IncompleteBlockHeader::from_bytes(cabecera).is_ok()
}

#[cfg(test)]
mod pruebas {
    use super::*;

    #[test]
    fn la_configuracion_es_la_declarada() {
        assert_eq!(
            hex::encode(configuracion().unwrap().to_bytes()),
            crate::CONFIG_HEX
        );
    }

    #[test]
    fn arbol_cte_igual_al_oficial() {
        let clave = [0x5A; 32];
        let porciones = 1 << 12;
        let oficial = MerkleTree::new(&vec![BYTE_CTE; porciones * PORCION], clave);
        let propio = ArbolCte::nuevo(&clave, porciones);
        assert_eq!(propio.raiz, oficial.root());
        for hojas in [
            vec![0],
            vec![1],
            vec![4095],
            vec![7, 8],
            vec![0, 1, 2, 3, 4094, 4095],
            (64..128).collect(),
            vec![3, 500, 501, 502, 2048, 4000],
        ] {
            let (a, b) = (oficial.get_multileaf_proof(&hojas), propio.prueba(&hojas));
            assert_eq!(a.siblings, b.siblings, "{hojas:?}");
            assert_eq!(a.leaf_indices, b.leaf_indices);
            assert_eq!(a.total_leaves, b.total_leaves);
            assert_eq!(a.root, b.root);
            assert_eq!(a.leaf_data, b.leaf_data);
        }
    }

    #[test]
    fn la_receta_cae_en_rango() {
        let a = matriz_a(&[7; 32], 256);
        let (cuerpo, cola) = a.split_at(a.len() - 1024);
        assert!(cuerpo.iter().all(|&b| b == 64));
        assert!(cola.iter().all(|&b| (-64..=63).contains(&(b as i8))));
        assert!(cola.iter().any(|&b| b != 64));
    }
}
