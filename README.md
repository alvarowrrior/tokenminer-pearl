# TokenMiner · motor de Pearl

Motor de minería de [Pearl](https://github.com/pearl-research-labs/pearl) para tarjetas NVIDIA.
Es la biblioteca que usa el programa de TokenMiner; aquí va sola, con sus pruebas.

Pearl no es una prueba de trabajo corriente: cada vuelta construye matrices enormes desde una
semilla, las multiplica en los tensor cores con enteros de 8 bits y acumulación de 32, y de ahí
salen las shares. Este motor hace ese producto en la GPU y reconstruye cada share con el código
oficial antes de entregarla.

## Qué hace

- **Un motor por GPU.** Todas reciben los mismos trabajos y sus shares salen por la misma conexión.
- **Dos variantes de kernel.** `Tma`, la rápida, es cubin de sm_120 (RTX 50). `CpAsync` es PTX de
  sm_80 que compila el driver, para las demás RTX. Las dos pasan una autocomprobación antes de
  empezar a minar.
- **Cada share se verifica.** Se reconstruye con el código oficial de Pearl (`zk-pow` y
  `pearl-blake3`) antes de salir, así que no se envía nada que no cuadre.
- **Límite de uso de la GPU** por pausas, para dejar sitio a otros programas.

## Requisitos

- Tarjeta NVIDIA con CUDA. La variante rápida necesita sm_120 (RTX 50); el resto usa la de sm_80.
- `nvcuda.dll`, que carga en tiempo de ejecución; no hace falta el kit de CUDA para compilar.
- Rust estable.

## Pruebas

```powershell
cargo test
```

Las que necesitan GPU o el verificador están marcadas `#[ignore]`. El verificador se compila
aparte y después se lanzan:

```powershell
cargo build --release --manifest-path tools/verifier/Cargo.toml
cargo test --release -- --ignored --test-threads=1
```

- `tests/oficial.rs` contrasta el cálculo en CPU contra el código oficial de Pearl, sin tocar la GPU.
- `tests/gpu.rs` hace una vuelta completa en la GPU y la compara con `pearl-verifier round`.
  Necesita la GPU libre.

`tools/verifier` es `pearl-verifier`, un programa aparte que no comparte código con el motor: usa
sólo el código oficial de Pearl que va en `vendor/`. Verifica shares (`serve`), calcula una vuelta
completa (`round`) y construye shares (`shares`). Las pruebas lo buscan en `tools/verifier/target/release`,
o en la ruta de la variable `PEARL_VERIFIER`.

## Kernels

`kernels/fuentes/` tiene las fuentes CUDA, y `kernels/` los binarios compilados.
`kernels/manifest.json` registra, para cada uno, de qué fuente sale, con qué opciones se compiló y
los SHA256 de la fuente y de la imagen, de modo que la compilación se puede reproducir y comprobar.

## Licencia

GPL-3.0-only. Puedes leer, modificar y redistribuir el código; si distribuyes una versión
modificada, tienes que publicar la tuya bajo la misma licencia.

La especificación que sigue este motor es la de
[pearl-research-labs/pearl](https://github.com/pearl-research-labs/pearl), licencia ISC, y usa
`plonky2` (Apache-2.0 y MIT). Detalle en [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) y en
`licenses/`.
