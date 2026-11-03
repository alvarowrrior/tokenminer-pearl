//! Contraste en CPU con el código oficial, a través de `pearl-verifier shares`. No toca la GPU.
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tokenminer_pearl::{cota_penalizada, job_key, CONFIG_HEX};

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

#[test]
#[ignore = "necesita pearl-verifier"]
fn clave_configuracion_y_cota_coinciden_con_el_oficial() {
    let mut hijo = Command::new(verificador())
        .args(["shares", "2"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("no se pudo lanzar pearl-verifier");
    let mut entrada = hijo.stdin.take().unwrap();
    let mut salida = BufReader::new(hijo.stdout.take().unwrap());
    let casos: [(&str, &str); 3] = [
        (
            "00",
            "000000000000000000000000000000000000000000000000000000000000ffff",
        ),
        (
            "a7",
            "000000000000000000000003fffc000000000000000000000000000000000000",
        ),
        (
            "5e",
            "000007ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        ),
    ];
    for (i, (b, objetivo)) in casos.iter().enumerate() {
        let cabecera = b.repeat(76);
        writeln!(
            entrada,
            "{}",
            json!({"op": "job", "id": i, "header": cabecera, "target": objetivo})
        )
        .unwrap();
        entrada.flush().unwrap();
        let mut linea = String::new();
        salida.read_line(&mut linea).unwrap();
        let r: Value = serde_json::from_str(&linea).unwrap();
        assert_eq!(r["ok"], json!(true), "{r}");
        assert_eq!(r["config"], json!(CONFIG_HEX));
        let h: [u8; 76] = hex::decode(&cabecera).unwrap().try_into().unwrap();
        assert_eq!(r["jobKey"], json!(hex::encode(job_key(&h))));
        let t: [u8; 32] = hex::decode(objetivo).unwrap().try_into().unwrap();
        let mut c = cota_penalizada(&t).unwrap();
        c.reverse();
        assert_eq!(r["bound"], json!(hex::encode(c)));
    }
    drop(entrada);
    hijo.wait().unwrap();
}
