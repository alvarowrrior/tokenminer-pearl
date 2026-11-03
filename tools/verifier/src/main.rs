//! Verificador de shares de Pearl con el código oficial (`zk-pow` y `pearl-blake3`, ISC), para contrastar el
//! motor. No comparte código con él: las matrices, las raíces, el ruido y las pruebas salen del código oficial.
//!
//! `pearl-verifier serve`: lee por stdin una línea JSON por share
//!   {"id":..,"header":"<76 bytes hex>","proof":"<base64 de PlainProof>","cert":3,"nbits":<u32 opcional>}
//!   y responde por stdout {"id":..,"ok":true} o {"id":..,"ok":false,"error":".."}.
//! `pearl-verifier round <header_hex> <m> <n> <k> <rank> <semilla_a> <semilla_b> <dir> [forma]`: vuelta completa
//!   (matrices, raíces, salado V3, semillas, ruido y hash de cada tesela) con el código oficial.
//! `pearl-verifier shares [forma]`: construye shares con el código oficial (ops `job` y `proof`, JSON por línea;
//!   ver `shares`).
//!
//! Formas: 1, k 4096, tesela 8×16 y matrices de la receta; 2 (la del motor), k 8192, tesela 12×12, A y Bᵀ
//! constantes en 64 salvo la última porción de A, que lleva la receta (ver `matriz_a`).
use anyhow::{Context, Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use primitive_types::U256;
use std::io::{BufRead, Write};
use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern};
use zk_pow::api::verify::verify_plain_proof;
use zk_pow::ffi::plain_proof::{MatrixMerkleProof, PlainProof, check_cert_version_eligible};

fn verify(header_hex: &str, proof_b64: &str, cert: u32, nbits: Option<u32>) -> Result<()> {
    let header =
        IncompleteBlockHeader::from_bytes(&hex::decode(header_hex).context("header hex")?)?;
    let proof =
        PlainProof::deserialize_compat(&STANDARD.decode(proof_b64.trim()).context("base64")?)?;
    let version = check_cert_version_eligible(cert, &proof)?;
    verify_plain_proof(&header, &proof, nbits, version.seed_derivation())
}

fn serve() -> Result<()> {
    let stdout = std::io::stdout();
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<serde_json::Value>(&line) {
            Err(e) => serde_json::json!({"id": null, "ok": false, "error": format!("json: {e}")}),
            Ok(req) => {
                let result = (|| -> Result<()> {
                    let header = req["header"].as_str().context("header")?;
                    let proof = req["proof"].as_str().context("proof")?;
                    let cert = req["cert"].as_u64().context("cert")? as u32;
                    let nbits = req["nbits"].as_u64().map(|v| v as u32);
                    verify(header, proof, cert, nbits)
                })();
                match result {
                    Ok(()) => serde_json::json!({"id": req["id"], "ok": true}),
                    Err(e) => {
                        serde_json::json!({"id": req["id"], "ok": false, "error": format!("{e:#}")})
                    }
                }
            }
        };
        let mut out = stdout.lock();
        writeln!(out, "{reply}")?;
        out.flush()?;
    }
    Ok(())
}

fn job_key(header: &IncompleteBlockHeader, config: &MiningConfiguration) -> [u8; 32] {
    let mut data = header.to_bytes().to_vec();
    data.extend_from_slice(&config.to_bytes());
    *blake3::hash(&data).as_bytes()
}

/// Forma del producto: k, tesela de warp WM×WN → h = WM/8 filas y w = 2·WN/8 columnas, bloque BM×BN.
#[derive(Clone, Copy)]
struct Forma {
    k: usize,
    h: usize,
    w: usize,
    bm: usize,
    bn: usize,
    cte: bool,
}

impl Forma {
    fn de(id: u64) -> Result<Forma> {
        match id {
            1 => Ok(Forma {
                k: 4096,
                h: 8,
                w: 16,
                bm: 128,
                bn: 256,
                cte: false,
            }),
            2 => Ok(Forma {
                k: 8192,
                h: 12,
                w: 12,
                bm: 192,
                bn: 192,
                cte: true,
            }),
            _ => bail!("forma desconocida"),
        }
    }
    /// Filas y columnas que trabaja el producto: los bloques enteros.
    fn usadas(&self, m: usize, n: usize) -> (usize, usize) {
        (m / self.bm * self.bm, n / self.bn * self.bn)
    }
}

/// Configuración de minado: filas g+{0,8,…}, columnas 2t+{0,1,8,9,…}.
fn config_forma(f: Forma, rank: usize) -> Result<MiningConfiguration> {
    let rows: Vec<u32> = (0..f.h as u32).map(|i| i * 8).collect();
    let cols: Vec<u32> = (0..f.w as u32 / 2)
        .flat_map(|i| [i * 8, i * 8 + 1])
        .collect();
    Ok(MiningConfiguration {
        common_dim: f.k as u32,
        rank: rank as u16,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&rows)?,
        cols_pattern: PeriodicPattern::from_list(&cols)?,
        moe: None,
    })
}

/// Receta de matrices con semilla propia por matriz: palabra i de la matriz por filas
/// = mezcla(i + semilla[i mod 8]), cuatro bytes por palabra, cada uno (b & 127) − 64.
fn matriz(seed: &[u32; 8], rows: usize, k: usize) -> Vec<u8> {
    let words = rows * k / 4;
    let mut out = vec![0u8; words * 4];
    receta(seed, &mut out, 0);
    out
}

/// Escribe en `out` la receta desde la palabra `desde`.
fn receta(seed: &[u32; 8], out: &mut [u8], desde: usize) {
    for (w, b) in out.chunks_exact_mut(4).enumerate() {
        let i = desde + w;
        let mut h = (i as u32).wrapping_add(seed[i % 8]);
        h = (h ^ (h >> 16)).wrapping_mul(0x7feb352d);
        h = (h ^ (h >> 15)).wrapping_mul(0x846ca68b);
        h ^= h >> 16;
        for j in 0..4 {
            b[j] = ((((h >> (j * 8)) & 127) as i32) - 64) as i8 as u8;
        }
    }
}

/// A de la vuelta. Forma 2: 64 en todos los bytes salvo la última porción de 1024 B, con la receta.
fn matriz_a(f: Forma, seed: &[u32; 8], m: usize) -> Vec<u8> {
    if !f.cte {
        return matriz(seed, m, f.k);
    }
    let mut a = vec![64u8; m * f.k];
    let desde = a.len() - 1024;
    receta(seed, &mut a[desde..], desde / 4);
    a
}

/// Bᵀ del trabajo. Forma 2: 64 en todos los bytes.
fn matriz_bt(f: Forma, seed: &[u32; 8], n: usize) -> Vec<u8> {
    if f.cte {
        vec![64u8; n * f.k]
    } else {
        matriz(seed, n, f.k)
    }
}

/// Las teselas de un patrón como `threads_partition` de `ffi/mine.rs` (privada en el oficial).
fn partition(pattern: &PeriodicPattern, total: usize) -> Vec<Vec<usize>> {
    let base: Vec<usize> = pattern.to_list().iter().map(|&i| i as usize).collect();
    (0..total)
        .filter(|&i| pattern.offset_is_valid(i as u32))
        .map(|o| base.iter().map(|&d| o + d).collect())
        .collect()
}

/// Vuelta completa con el código oficial, para contrastar la GPU.
/// Imprime JSON: config, job_key, raíces, raíces saladas, semillas y el hash de cada tesela (fila base, columna
/// base). Escribe A' y B'ᵀ en `<dir>/ap.bin` y `<dir>/btp.bin`.
fn round_cli(args: &[String]) -> Result<()> {
    if args.len() < 8 {
        bail!(
            "uso: pearl-verifier round <header_hex> <m> <n> <k> <rank> <semilla_a_hex32> <semilla_b_hex32> <dir> [forma]"
        );
    }
    use zk_pow::api::proof::SeedDerivation;
    let header = IncompleteBlockHeader::from_bytes(&hex::decode(&args[0]).context("header")?)?;
    let (m, n, k, rank): (usize, usize, usize, usize) = (
        args[1].parse()?,
        args[2].parse()?,
        args[3].parse()?,
        args[4].parse()?,
    );
    let seed = |s: &str| -> Result<[u32; 8]> {
        let b = hex::decode(s).context("semilla")?;
        if b.len() != 32 {
            bail!("semilla de 32 bytes")
        }
        Ok(std::array::from_fn(|i| {
            u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
        }))
    };
    let (seed_a, seed_b) = (seed(&args[5])?, seed(&args[6])?);
    let forma = Forma::de(args.get(8).map_or(Ok(1), |s| s.parse())?)?;
    if forma.k != k {
        bail!("k no es el de la forma")
    }
    let (mu, nu) = forma.usadas(m, n);
    let config = config_forma(forma, rank)?;
    let key = job_key(&header, &config);

    let a = matriz_a(forma, &seed_a, m);
    let bt = matriz_bt(forma, &seed_b, n);
    // Como `compute_commitment_hash_with_offsets` (privada): raíz = BLAKE3 con clave del relleno,
    // salado V3, y la cadena de semillas de `PublicProofParams::commitment_hash`.
    let root_a = pearl_blake3::blake3_digest(&pearl_blake3::pad_to_chunk_boundary(&a), Some(key));
    let root_b = pearl_blake3::blake3_digest(&pearl_blake3::pad_to_chunk_boundary(&bt), Some(key));
    let (bound_a, bound_b) =
        SeedDerivation::Salted.bind_roots(&root_a, &root_b, m as u32, n as u32);
    let b_seed = pearl_blake3::blake3_digest(&[&key[..], &bound_b[..]].concat(), None);
    let a_seed = pearl_blake3::blake3_digest(&[&b_seed[..], &bound_a[..]].concat(), None);

    let noise = zk_pow::circuit::pearl_noise::compute_noise_for_indices(
        k,
        rank,
        (b_seed, a_seed),
        &(0..mu).collect::<Vec<_>>(),
        &(0..nu).collect::<Vec<_>>(),
    );
    let add = |x: &[u8], rows: &Vec<Vec<i8>>| -> Vec<i8> {
        (0..rows.len() * k)
            .map(|i| (x[i] as i8 as i32 + rows[i / k][i % k] as i32) as i8)
            .collect()
    };
    let ap = add(&a, &noise.a);
    let btp = add(&bt, &noise.b);

    // Jackpots de cada tesela, como el bucle de `try_mine_one`.
    use zk_pow::circuit::pearl_program::{JACKPOT_SIZE, LROT_PER_TILE};
    let mut tiles = Vec::new();
    for rows in partition(&config.rows_pattern, mu) {
        for cols in partition(&config.cols_pattern, nu) {
            let mut acc = vec![0i32; rows.len() * cols.len()];
            let mut jackpot = [0u32; 16];
            for ll in (rank..=k).step_by(rank) {
                for (u, &r) in rows.iter().enumerate() {
                    for (v, &c) in cols.iter().enumerate() {
                        let (ar, bc) = (&ap[r * k..r * k + k], &btp[c * k..c * k + k]);
                        let mut s = acc[u * cols.len() + v];
                        for l in ll - rank..ll {
                            s = s.wrapping_add(ar[l] as i32 * bc[l] as i32);
                        }
                        acc[u * cols.len() + v] = s;
                    }
                }
                let x = acc.iter().fold(0u32, |a, &v| a ^ v as u32);
                let tid = (ll / rank - 1) % JACKPOT_SIZE;
                jackpot[tid] = jackpot[tid].rotate_left(LROT_PER_TILE) ^ x;
            }
            let h = zk_pow::api::proof_utils::compute_jackpot_hash(&jackpot, a_seed);
            tiles.push(serde_json::json!([rows[0], cols[0], hex::encode(h)]));
        }
    }
    let dir = std::path::Path::new(&args[7]);
    std::fs::create_dir_all(dir)?;
    let as_u8 = |v: &[i8]| v.iter().map(|&x| x as u8).collect::<Vec<u8>>();
    std::fs::write(dir.join("ap.bin"), as_u8(&ap))?;
    std::fs::write(dir.join("btp.bin"), as_u8(&btp))?;
    println!(
        "{}",
        serde_json::json!({
            "config": hex::encode(config.to_bytes()), "jobKey": hex::encode(key),
            "rootA": hex::encode(root_a), "rootB": hex::encode(root_b),
            "boundA": hex::encode(bound_a), "boundB": hex::encode(bound_b),
            "bSeed": hex::encode(b_seed), "aSeed": hex::encode(a_seed),
            "tiles": tiles,
        })
    );
    Ok(())
}

/// Constructor de shares con el código oficial. Una línea JSON por petición:
///   {"op":"job","id":..,"header":"..","target":"<BE hex>"} → {"jobKey","config","bound"(cota penalizada, BE hex)}
///   {"op":"proof","id":..,"header":"..","m":..,"n":..,"seedA":"<32 B hex>","seedB":"..","row0":..,"col0":..}
///     → {"proof":"<base64 de PlainProof>"}
/// A (de la vuelta) y Bᵀ (del trabajo) se reconstruyen con la receta; el árbol de Bᵀ se guarda mientras no
/// cambien su semilla, n o la clave de trabajo, y el de A mientras no cambie la vuelta.
fn shares(forma: Forma) -> Result<()> {
    const RANK: usize = 128;
    let k = forma.k;
    let config = config_forma(forma, RANK)?;
    // h·w · (k/r) · 128 (penalized_target_bound).
    let factor = U256::from(forma.h * forma.w * (k / RANK) * 128);
    let (pm, pn) = (8 * forma.h, 4 * forma.w); // periodos de los patrones
    let mut cache_b: Option<(String, pearl_blake3::MerkleTree)> = None;
    let mut cache_a: Option<(String, pearl_blake3::MerkleTree)> = None;
    let seed = |v: &serde_json::Value, name: &str| -> Result<[u32; 8]> {
        let b = hex::decode(v[name].as_str().context(name.to_string())?)?;
        if b.len() != 32 {
            bail!("{name} de 32 bytes")
        }
        Ok(std::array::from_fn(|i| {
            u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
        }))
    };
    let part = |tree: &pearl_blake3::MerkleTree, rows: Vec<usize>, dims: (usize, usize)| {
        let leaves = pearl_blake3::MerkleTree::compute_leaf_indices_from_rows(&rows, dims);
        MatrixMerkleProof {
            proof: tree.get_multileaf_proof(&leaves),
            row_indices: rows,
        }
    };
    let stdout = std::io::stdout();
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let req: serde_json::Value = serde_json::from_str(&line)?;
        let result = (|| -> Result<serde_json::Value> {
            let header = IncompleteBlockHeader::from_bytes(&hex::decode(
                req["header"].as_str().context("header")?,
            )?)?;
            let key = job_key(&header, &config);
            match req["op"].as_str() {
                Some("job") => {
                    let target = U256::from_big_endian(&hex::decode(
                        req["target"].as_str().context("target")?,
                    )?);
                    let bound = target
                        .checked_mul(factor)
                        .context("objetivo demasiado fácil")?;
                    Ok(
                        serde_json::json!({"jobKey": hex::encode(key), "config": hex::encode(config.to_bytes()),
                        "bound": format!("{bound:064x}")}),
                    )
                }
                Some("proof") => {
                    let m = req["m"].as_u64().context("m")? as usize;
                    let n = req["n"].as_u64().context("n")? as usize;
                    let (row0, col0) = (
                        req["row0"].as_u64().context("row0")? as usize,
                        req["col0"].as_u64().context("col0")? as usize,
                    );
                    let (mu, nu) = forma.usadas(m, n);
                    if row0 % pm >= 8 || row0 + 8 * (forma.h - 1) >= mu {
                        bail!("fila base fuera de patrón")
                    }
                    if col0 % pn >= 8 || col0 % 2 != 0 || col0 + 8 * (forma.w / 2 - 1) + 1 >= nu {
                        bail!("columna base fuera de patrón")
                    }
                    let (sa, sb) = (seed(&req, "seedA")?, seed(&req, "seedB")?);
                    let tag_b = format!("{}:{n}:{}", req["seedB"], hex::encode(key));
                    if cache_b.as_ref().map(|c| &c.0) != Some(&tag_b) {
                        let bt = matriz_bt(forma, &sb, n);
                        cache_b = Some((
                            tag_b,
                            pearl_blake3::MerkleTree::new(
                                &pearl_blake3::pad_to_chunk_boundary(&bt),
                                key,
                            ),
                        ));
                    }
                    let tag_a = format!("{}:{m}:{}", req["seedA"], hex::encode(key));
                    if cache_a.as_ref().map(|c| &c.0) != Some(&tag_a) {
                        let a = matriz_a(forma, &sa, m);
                        cache_a = Some((
                            tag_a,
                            pearl_blake3::MerkleTree::new(
                                &pearl_blake3::pad_to_chunk_boundary(&a),
                                key,
                            ),
                        ));
                    }
                    let rows: Vec<usize> = (0..forma.h).map(|i| row0 + 8 * i).collect();
                    let cols: Vec<usize> = (0..forma.w / 2)
                        .flat_map(|i| [col0 + 8 * i, col0 + 8 * i + 1])
                        .collect();
                    let proof = PlainProof {
                        m,
                        n,
                        k,
                        noise_rank: RANK,
                        a: part(&cache_a.as_ref().unwrap().1, rows, (m, k)),
                        bt: part(&cache_b.as_ref().unwrap().1, cols, (n, k)),
                        moe: None,
                    };
                    Ok(serde_json::json!({"proof": STANDARD.encode(bincode::serialize(&proof)?)}))
                }
                _ => bail!("op desconocida"),
            }
        })();
        let reply = match result {
            Ok(mut v) => {
                v["id"] = req["id"].clone();
                v["ok"] = true.into();
                v
            }
            Err(e) => serde_json::json!({"id": req["id"], "ok": false, "error": format!("{e:#}")}),
        };
        let mut out = stdout.lock();
        writeln!(out, "{reply}")?;
        out.flush()?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("serve") => serve(),
        Some("round") => round_cli(&args[1..]),
        Some("shares") => shares(Forma::de(args.get(1).map_or(Ok(1), |s| s.parse())?)?),
        _ => bail!("uso: pearl-verifier serve | round ... | shares [forma]"),
    }
}
