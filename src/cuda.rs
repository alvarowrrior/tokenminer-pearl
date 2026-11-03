//! Capa fina sobre `nvcuda.dll`, cargada en tiempo de ejecución. Todas las llamadas deben hacerse
//! desde el hilo que creó `Cuda`.
use libloading::Library;
use std::ffi::{c_int, c_uint, c_void, CString};

pub type Ptr = *mut c_void;

type Resultado<T> = Result<T, String>;

macro_rules! api {
    ($($campo:ident : $simbolo:literal = fn($($t:ty),*);)*) => {
        struct Api { $($campo: unsafe extern "system" fn($($t),*) -> c_int,)* }
        impl Api {
            unsafe fn cargar(lib: &Library) -> Resultado<Api> {
                Ok(Api { $($campo: *lib.get($simbolo).map_err(|e| format!("{}: {e}", stringify!($campo)))?,)* })
            }
        }
    };
}

api! {
    init: b"cuInit\0" = fn(c_uint);
    device_get: b"cuDeviceGet\0" = fn(*mut c_int, c_int);
    device_count: b"cuDeviceGetCount\0" = fn(*mut c_int);
    device_luid: b"cuDeviceGetLuid\0" = fn(*mut u8, *mut c_uint, c_int);
    device_capability: b"cuDeviceComputeCapability\0" = fn(*mut c_int, *mut c_int, c_int);
    device_attr: b"cuDeviceGetAttribute\0" = fn(*mut c_int, c_int, c_int);
    ctx_create: b"cuCtxCreate_v2\0" = fn(*mut Ptr, c_uint, c_int);
    ctx_destroy: b"cuCtxDestroy_v2\0" = fn(Ptr);
    ctx_sync: b"cuCtxSynchronize\0" = fn();
    module_load: b"cuModuleLoadData\0" = fn(*mut Ptr, *const c_void);
    module_unload: b"cuModuleUnload\0" = fn(Ptr);
    module_fn: b"cuModuleGetFunction\0" = fn(*mut Ptr, Ptr, *const std::ffi::c_char);
    func_attr: b"cuFuncSetAttribute\0" = fn(Ptr, c_int, c_int);
    mem_alloc: b"cuMemAlloc_v2\0" = fn(*mut u64, usize);
    mem_free: b"cuMemFree_v2\0" = fn(u64);
    mem_info: b"cuMemGetInfo_v2\0" = fn(*mut usize, *mut usize);
    htod: b"cuMemcpyHtoD_v2\0" = fn(u64, *const c_void, usize);
    dtoh: b"cuMemcpyDtoH_v2\0" = fn(*mut c_void, u64, usize);
    memset: b"cuMemsetD8_v2\0" = fn(u64, u8, usize);
    tensor_map: b"cuTensorMapEncodeTiled\0" = fn(*mut c_void, c_int, c_uint, u64, *const u64, *const u64,
                                                 *const c_uint, *const c_uint, c_int, c_int, c_int, c_int);
    launch: b"cuLaunchKernel\0" = fn(Ptr, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Ptr,
                                     *mut *mut c_void, *mut *mut c_void);
    stream_create: b"cuStreamCreate\0" = fn(*mut Ptr, c_uint);
    stream_destroy: b"cuStreamDestroy_v2\0" = fn(Ptr);
    event_create: b"cuEventCreate\0" = fn(*mut Ptr, c_uint);
    event_destroy: b"cuEventDestroy_v2\0" = fn(Ptr);
    event_record: b"cuEventRecord\0" = fn(Ptr, Ptr);
    event_sync: b"cuEventSynchronize\0" = fn(Ptr);
}

fn chk(nombre: &str, codigo: c_int) -> Resultado<()> {
    if codigo == 0 {
        Ok(())
    } else {
        Err(format!("{nombre}: error CUDA {codigo}"))
    }
}

/// Descriptor TMA, que el kernel recibe por valor.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct TmaMap(pub [u8; 128]);

/// Un parámetro de kernel.
#[derive(Clone, Copy)]
pub enum Arg {
    Mem(u64),
    U32(u32),
    I32(i32),
    Map(TmaMap),
}

pub struct Cuda {
    api: Api,
    dev: c_int,
    ctx: Ptr,
    mods: Vec<Ptr>,
    allocs: Vec<u64>,
    streams: Vec<Ptr>,
    events: Vec<Ptr>,
    _lib: Library,
}

unsafe fn abrir() -> Resultado<(Library, Api)> {
    let lib = Library::new("nvcuda.dll").map_err(|e| format!("nvcuda.dll: {e}"))?;
    let api = Api::cargar(&lib)?;
    chk("cuInit", (api.init)(0))?;
    Ok((lib, api))
}

/// El LUID de cada GPU, en el orden de CUDA y escrito como en los contadores de Windows: `0x<alto>_0x<bajo>`.
pub fn gpus() -> Resultado<Vec<String>> {
    unsafe {
        let (_lib, api) = abrir()?;
        let mut cuantas: c_int = 0;
        chk("cuDeviceGetCount", (api.device_count)(&mut cuantas))?;
        (0..cuantas)
            .map(|i| {
                let (mut dev, mut luid, mut mascara) = (0, [0u8; 8], 0);
                chk("cuDeviceGet", (api.device_get)(&mut dev, i))?;
                chk(
                    "cuDeviceGetLuid",
                    (api.device_luid)(luid.as_mut_ptr(), &mut mascara, dev),
                )?;
                let [b0, b1, b2, b3, a0, a1, a2, a3] = luid;
                Ok(format!(
                    "0x{:08X}_0x{:08X}",
                    u32::from_le_bytes([a0, a1, a2, a3]),
                    u32::from_le_bytes([b0, b1, b2, b3])
                ))
            })
            .collect()
    }
}

impl Cuda {
    /// Contexto en la GPU `dispositivo`, en el orden de CUDA.
    pub fn en(dispositivo: i32) -> Resultado<Cuda> {
        unsafe {
            let (lib, api) = abrir()?;
            let mut dev: c_int = 0;
            chk("cuDeviceGet", (api.device_get)(&mut dev, dispositivo))?;
            let mut ctx: Ptr = std::ptr::null_mut();
            // CU_CTX_SCHED_BLOCKING_SYNC = 4: la CPU duerme mientras espera a la GPU.
            chk("cuCtxCreate", (api.ctx_create)(&mut ctx, 4, dev))?;
            Ok(Cuda {
                api,
                dev,
                ctx,
                mods: vec![],
                allocs: vec![],
                streams: vec![],
                events: vec![],
                _lib: lib,
            })
        }
    }

    /// Arquitectura de la GPU: (12, 0) es sm_120.
    pub fn arquitectura(&self) -> Resultado<(i32, i32)> {
        let (mut mayor, mut menor) = (0, 0);
        chk("cuDeviceComputeCapability", unsafe {
            (self.api.device_capability)(&mut mayor, &mut menor, self.dev)
        })?;
        Ok((mayor, menor))
    }

    /// Memoria compartida máxima por bloque (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 81).
    pub fn compartida_maxima(&self) -> Resultado<u32> {
        let mut v = 0;
        chk("cuDeviceGetAttribute", unsafe {
            (self.api.device_attr)(&mut v, 81, self.dev)
        })?;
        Ok(v as u32)
    }

    /// Carga un cubin, o PTX que compila el driver. Se copia a un búfer alineado a 8, porque `include_bytes!` no
    /// garantiza alineación, y acabado en cero, que es lo que pide el PTX.
    pub fn cargar(&mut self, imagen: &[u8]) -> Resultado<Ptr> {
        let mut alineada = vec![0u64; imagen.len().div_ceil(8) + 1];
        unsafe {
            std::ptr::copy_nonoverlapping(
                imagen.as_ptr(),
                alineada.as_mut_ptr() as *mut u8,
                imagen.len(),
            );
            let mut m: Ptr = std::ptr::null_mut();
            chk(
                "cuModuleLoadData",
                (self.api.module_load)(&mut m, alineada.as_ptr() as *const c_void),
            )?;
            self.mods.push(m);
            Ok(m)
        }
    }

    pub fn funcion(&self, modulo: Ptr, nombre: &str, shared: u32) -> Resultado<Ptr> {
        let c_nombre = CString::new(nombre).map_err(|e| e.to_string())?;
        unsafe {
            let mut f: Ptr = std::ptr::null_mut();
            chk(
                nombre,
                (self.api.module_fn)(&mut f, modulo, c_nombre.as_ptr()),
            )?;
            if shared > 48 * 1024 {
                // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
                chk(
                    "cuFuncSetAttribute",
                    (self.api.func_attr)(f, 8, shared as c_int),
                )?;
            }
            Ok(f)
        }
    }

    pub fn malloc(&mut self, bytes: usize) -> Resultado<u64> {
        let mut p = 0u64;
        chk("cuMemAlloc", unsafe { (self.api.mem_alloc)(&mut p, bytes) })?;
        self.allocs.push(p);
        Ok(p)
    }

    /// (libre, total) de memoria de la GPU en bytes.
    pub fn memoria(&self) -> Resultado<(usize, usize)> {
        let (mut libre, mut total) = (0usize, 0usize);
        chk("cuMemGetInfo", unsafe {
            (self.api.mem_info)(&mut libre, &mut total)
        })?;
        Ok((libre, total))
    }

    pub fn htod<T: Copy>(&self, destino: u64, datos: &[T]) -> Resultado<()> {
        chk("cuMemcpyHtoD", unsafe {
            (self.api.htod)(
                destino,
                datos.as_ptr() as *const c_void,
                std::mem::size_of_val(datos),
            )
        })
    }

    pub fn dtoh<T: Copy>(&self, datos: &mut [T], origen: u64) -> Resultado<()> {
        chk("cuMemcpyDtoH", unsafe {
            (self.api.dtoh)(
                datos.as_mut_ptr() as *mut c_void,
                origen,
                std::mem::size_of_val(datos),
            )
        })
    }

    pub fn a_cero(&self, mem: u64, bytes: usize) -> Resultado<()> {
        chk("cuMemsetD8", unsafe { (self.api.memset)(mem, 0, bytes) })
    }

    /// Descriptor TMA 2D de una matriz int8: `inner` bytes por fila, `outer` filas, tesela box_inner×box_outer.
    /// Swizzle de 128 B (`box_inner` = 128, como `swz()` del núcleo) y promoción a L2 de 128 B.
    pub fn tensor_map(
        &self,
        mem: u64,
        inner: u64,
        outer: u64,
        box_inner: u32,
        box_outer: u32,
    ) -> Resultado<TmaMap> {
        assert_eq!(box_inner, 128, "el núcleo sólo se compila con BK = 128");
        let mut mapa = TmaMap([0; 128]);
        let dim = [inner, outer]; // el más interno primero
        let strides = [inner]; // bytes entre filas
        let caja = [box_inner, box_outer];
        let paso = [1u32, 1];
        // UINT8 = 0, rango 2, sin intercalado = 0, swizzle 128 B = 3, promoción 128 B = 2, sin relleno = 0.
        chk("cuTensorMapEncodeTiled", unsafe {
            (self.api.tensor_map)(
                &mut mapa as *mut TmaMap as *mut c_void,
                0,
                2,
                mem,
                dim.as_ptr(),
                strides.as_ptr(),
                caja.as_ptr(),
                paso.as_ptr(),
                0,
                3,
                2,
                0,
            )
        })?;
        Ok(mapa)
    }

    /// Stream que no se sincroniza con el stream nulo (CU_STREAM_NON_BLOCKING).
    pub fn stream(&mut self) -> Resultado<Ptr> {
        let mut s: Ptr = std::ptr::null_mut();
        chk("cuStreamCreate", unsafe {
            (self.api.stream_create)(&mut s, 1)
        })?;
        self.streams.push(s);
        Ok(s)
    }

    /// Evento con espera bloqueante (CU_EVENT_BLOCKING_SYNC): esperarlo no gira la CPU.
    pub fn evento(&mut self) -> Resultado<Ptr> {
        let mut e: Ptr = std::ptr::null_mut();
        chk("cuEventCreate", unsafe {
            (self.api.event_create)(&mut e, 1)
        })?;
        self.events.push(e);
        Ok(e)
    }

    pub fn registrar(&self, e: Ptr, s: Ptr) -> Resultado<()> {
        chk("cuEventRecord", unsafe { (self.api.event_record)(e, s) })
    }

    pub fn esperar(&self, e: Ptr) -> Resultado<()> {
        chk("cuEventSynchronize", unsafe { (self.api.event_sync)(e) })
    }

    pub fn sincronizar(&self) -> Resultado<()> {
        chk("cuCtxSynchronize", unsafe { (self.api.ctx_sync)() })
    }

    /// Lanza un kernel de `grid` × 1 × 1 bloques de `block` × 1 × 1 hilos.
    pub fn lanzar(
        &self,
        f: Ptr,
        grid: u32,
        block: u32,
        args: &[Arg],
        shared: u32,
        s: Ptr,
    ) -> Resultado<()> {
        // Cada entrada apunta al valor del parámetro; el driver los copia durante la llamada.
        let mut valores: Vec<Arg> = args.to_vec();
        let mut punteros: Vec<*mut c_void> = valores
            .iter_mut()
            .map(|a| match a {
                Arg::Mem(v) => v as *mut u64 as *mut c_void,
                Arg::U32(v) => v as *mut u32 as *mut c_void,
                Arg::I32(v) => v as *mut i32 as *mut c_void,
                Arg::Map(v) => v as *mut TmaMap as *mut c_void,
            })
            .collect();
        chk("cuLaunchKernel", unsafe {
            (self.api.launch)(
                f,
                grid,
                1,
                1,
                block,
                1,
                1,
                shared,
                s,
                punteros.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
    }
}

impl Drop for Cuda {
    fn drop(&mut self) {
        unsafe {
            (self.api.ctx_sync)();
            for &e in &self.events {
                (self.api.event_destroy)(e);
            }
            for &s in &self.streams {
                (self.api.stream_destroy)(s);
            }
            for &p in &self.allocs {
                (self.api.mem_free)(p);
            }
            for &m in &self.mods {
                (self.api.module_unload)(m);
            }
            (self.api.ctx_destroy)(self.ctx);
        }
    }
}
