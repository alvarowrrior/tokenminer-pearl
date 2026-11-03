// Núcleo rápido de Pearl para sm_120 (RTX 50).
// Producto A'·B'ᵀ int8 con acumulación int32 en tensor cores (mma.sync m16n8k32),
// plegado XOR por tramo de 128, jackpot de 16 palabras y BLAKE3 con clave en el epílogo.
// Código propio de TokenMiner. La especificación sigue a pearl-research-labs/pearl
// (zk-pow/src/ffi/mine.rs, try_mine_one), licencia ISC.
//
// Disposición: A' es m×k por filas; B' se guarda como B'ᵀ, n×k por filas (columnas de B contiguas).
// Cada hilo es dueño de una tesela de hash completa: filas g+{0,8,…,56} y columnas
// 2t+{0,1,8,9,…,56,57} de su tesela de warp 64×64 (g = lane/4, t = lane%4).
// Patrones oficiales: rows [0,8,16,24,32,40,48,56] (periodo 64), cols [0,1,8,9,…,56,57] (periodo 64).

#ifndef PL_K
#define PL_K 4096
#endif
#ifndef PL_STAGES
#define PL_STAGES 4
#endif
#ifndef PL_MODE
#define PL_MODE 1   // 0: escribe los 16 jackpots; 1: escribe un XOR por tesela; 2: BLAKE3 + comparación
#endif

#ifndef PL_BM
#define PL_BM 128
#endif
#ifndef PL_BN
#define PL_BN 256
#endif
#ifndef PL_MINB
#define PL_MINB 1
#endif
#ifndef PL_FOLD_TREE
#define PL_FOLD_TREE 1
#endif
#ifndef PL_TMA
#define PL_TMA 0   // 1: carga con cp.async.bulk.tensor (TMA) y mbarrier
#endif
#ifndef PL_TMA_EMPTY
#define PL_TMA_EMPTY 0   // con PL_TMA, en vez de __syncthreads: 1 = barreras «vacío» por etapa (sólo espera
                         // el productor, hilo 0); 2 = contador por etapa y el último warp que la suelta pide la siguiente
#endif
#ifndef PL_CLUSTER
#define PL_CLUSTER 0   // 2: clúster de 2 bloques con la misma tesela de B; cada uno pide la mitad por multicast
#endif
#ifndef PL_TILED
#define PL_TILED 0   // con PL_TMA: A' y B' guardadas por teselas de carga (bloque, etapa, fila) y no por filas
#endif
#ifndef PL_GA
#define PL_GA 0   // recorrido en olas de PL_GA × PL_GB bloques: los bloques de una ola arrancan a la vez y
#endif            // comparten en L2 las teselas de A y de B de cada etapa. 0: m-bloque más rápido.
#ifndef PL_GB
#define PL_GB 0
#endif
#ifndef PL_MMA_U
#define PL_MMA_U 0   // 1: mma sin signo; con A' y B' en [1, 127] el resultado es el mismo
#endif
#if PL_MMA_U
#define PL_MMA "s32.u8.u8.s32"
#else
#define PL_MMA "s32.s8.s8.s32"
#endif
#ifndef PL_ISSUE_EARLY
#define PL_ISSUE_EARLY 0   // con PL_TMA_EMPTY=1: el productor pide la etapa tras su último ldmatrix, no al final
#endif
#ifndef PL_BK
#define PL_BK 64
#endif
#ifndef PL_WM
#define PL_WM 64
#endif
#ifndef PL_WN
#define PL_WN 64
#endif
#define BM PL_BM
#define BN PL_BN
#define BK PL_BK
#define CHUNKS (BK / 16)
#define SWZ_SH (BK == 128 ? 0 : (BK == 64 ? 1 : 2))
#define KSTEPS (BK / 32)
#define KT_PER_TRAMO (128 / BK)
#define WM PL_WM
#define WN PL_WN
#define WARPS_N (BN / WN)
#define NTHREADS ((BM / WM) * (BN / WN) * 32)
#define MT (WM / 16)
#define NTL (WN / 8)
#define STAGE_A (BM * BK)
#define STAGE_B (BN * BK)
#define STAGE_BYTES (STAGE_A + STAGE_B)
#define TRAMOS (PL_K / 128)

typedef unsigned int u32;
typedef unsigned long long u64;

static_assert(!PL_CLUSTER || (PL_CLUSTER == 2 && PL_TMA && PL_TMA_EMPTY == 1 && !PL_ISSUE_EARLY),
              "PL_CLUSTER=2 exige PL_TMA y PL_TMA_EMPTY=1");
static_assert(!PL_TILED || (PL_TMA && BK == 128 && !PL_CLUSTER), "PL_TILED exige PL_TMA, BK = 128 y sin clúster");
static_assert(PL_K % 2048 == 0, "k/128 debe ser múltiplo de 16");
static_assert(BK == 32 || BK == 64 || BK == 128, "BK debe ser 32, 64 o 128");
// Tesela de hash: 2·MT filas × 2·NTL columnas = WM·WN/32 elementos, entre 32 y 256 (regla oficial).
static_assert(WM % 16 == 0 && WN % 16 == 0, "WM y WN múltiplos de 16 (NTL par)");
static_assert(WM * WN / 32 >= 32 && WM * WN / 32 <= 256, "h·w fuera de [32,256]");

__device__ __forceinline__ u32 smem_u32(const void* p) {
    return (u32)__cvta_generic_to_shared(p);
}

// Desplazamiento con swizzle de la fila `row`, trozo de 16 B `chunk` (0..CHUNKS-1), filas de BK bytes.
// El trozo se permuta con los bits de fila que cambian cada 128 B, para que las 8 filas que lee un
// ldmatrix caigan en los 32 bancos exactamente una vez.
__device__ __forceinline__ u32 swz(u32 row, u32 chunk) {
    return row * BK + ((chunk ^ ((row >> SWZ_SH) & (CHUNKS - 1))) << 4);
}

__device__ __forceinline__ void cp_async16(u32 dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

#if PL_TMA
// Coordenadas de TMA de la etapa kc de A' y de B'. Por filas: (kc, fila base del bloque). Por teselas,
// el tensor son filas de BK bytes en orden (bloque, etapa, fila), así que cada carga son BM·BK o BN·BK
// bytes contiguos en vez de BM o BN trozos separados por k. Sólo cambia dónde se guardan A' y B'.
#if PL_TILED
#define TA(kc) 0, ((bm * (PL_K / BK) + (kc) / BK) * BM)
#define TB(kc) 0, ((bn * (PL_K / BK) + (kc) / BK) * BN)
#else
#define TA(kc) (kc), m0
#define TB(kc) (kc), n0
#endif
// ---------------------------------------------------------------- TMA (cp.async.bulk.tensor)
// Un hilo pide la tesela entera; el hardware hace las direcciones, el swizzle de 128 B y la
// escritura a compartida, y avisa por mbarrier. Sustituye las NTHREADS·(APASS+BPASS) cp.async.
struct alignas(64) TmaMap { unsigned char raw[128]; };

__device__ __forceinline__ void tma_tile(u32 dst, const TmaMap* map, int c_inner, int c_outer,
                                         u32 mbar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        ::"r"(dst), "l"((unsigned long long)map), "r"(c_inner), "r"(c_outer), "r"(mbar)
        : "memory");
}
__device__ __forceinline__ void mbar_init(u32 mb, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(mb), "r"(count) : "memory");
}
// Llega al mbarrier y anuncia los bytes de la transacción a la vez: sin el `arrive` la cuenta
// de llegadas nunca se satisface y el `try_wait` gira para siempre.
__device__ __forceinline__ void mbar_expect(u32 mb, u32 bytes) {
    // El estado de vuelta no se usa: sumidero `_`, o el compilador lo guarda en memoria local.
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(mb), "r"(bytes) : "memory");
}
// Llegada simple (un warp avisa de que ya no lee la etapa).
__device__ __forceinline__ void mbar_arrive(u32 mb) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(mb) : "memory");
}
__device__ __forceinline__ void mbar_wait(u32 mb, u32 parity) {
    asm volatile("{\n\t.reg .pred p;\n\tPLW%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                 "@!p bra PLW%=;\n\t}\n" ::"r"(mb), "r"(parity) : "memory");
}
#if PL_CLUSTER
// ---------------------------------------------------------------- clúster de 2 bloques
// Los dos bloques tienen la misma n0 (bloques consecutivos de la rejilla, m más rápida) y la misma
// tesela de B. Cada uno pide su mitad de filas de B con multicast a los dos, en el mismo sitio de
// compartida, y avisa al mbarrier «lleno» de la misma etapa en los dos. La etapa sólo se puede
// recargar cuando la han soltado los warps de los dos bloques: «vacío» cuenta 16 llegadas.
__device__ __forceinline__ u32 cluster_rank() {
    u32 r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
    return r;
}
__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
// Dirección de la misma posición de compartida en el bloque `rank` del clúster.
__device__ __forceinline__ u32 peer_addr(u32 a, u32 rank) {
    u32 r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(rank));
    return r;
}
__device__ __forceinline__ void mbar_arrive_remote(u32 cluster_addr) {
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n" ::"r"(cluster_addr) : "memory");
}
__device__ __forceinline__ void mbar_wait_cluster(u32 mb, u32 parity) {
    asm volatile("{\n\t.reg .pred p;\n\tPLC%=: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n\t"
                 "@!p bra PLC%=;\n\t}\n" ::"r"(mb), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_tile_mc(u32 dst, const TmaMap* map, int c_inner, int c_outer, u32 mbar,
                                            unsigned short mask) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%2, %3}], [%4], %5;\n"
        ::"r"(dst), "l"((unsigned long long)map), "r"(c_inner), "r"(c_outer), "r"(mbar), "h"(mask)
        : "memory");
}
#endif
// Tesela de B de la etapa en `dst`: entera, o con clúster la mitad de este bloque para los dos.
__device__ __forceinline__ void tma_b(u32 dst, const TmaMap* map, int kc, int n0, u32 mbar) {
#if PL_CLUSTER && !defined(PL_PROBE_CLUSTER_NOMC)   // sonda: clúster sin multicast, cada bloque su B
    const u32 r = cluster_rank();
    tma_tile_mc(dst + r * (BN / 2) * BK, map, kc, n0 + (int)r * (BN / 2), mbar, (unsigned short)3);
#else
    tma_tile(dst, map, kc, n0, mbar);
#endif
}
#endif

__device__ __forceinline__ void ldsm_x4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void mma_s8(int* c, const u32* a, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col." PL_MMA " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Carga de una etapa (BM×BK de A y BN×BK de B) con cp.async de 16 B.
// Cada hilo carga el trozo `schunk` de las filas srow, srow+RPP, srow+2·RPP… Las filas que le
// tocan están a una distancia constante, así que `pa`/`pb` (una pareja de registros cada uno,
// que avanza BK por etapa) más desplazamientos inmediatos basta: sin recalcular direcciones.
#define RPP (NTHREADS / CHUNKS)   // filas por pasada
#define APASS (STAGE_A / 16 / NTHREADS)
#define BPASS (STAGE_B / 16 / NTHREADS)
static_assert(RPP % CHUNKS == 0, "RPP múltiplo de CHUNKS: el swizzle debe repetirse por pasada");

__device__ __forceinline__ void load_stage(u32 sbase, const signed char* __restrict__ pa,
                                           const signed char* __restrict__ pb, u32 srow, u32 schunk) {
#pragma unroll
    for (int i = 0; i < APASS; ++i)
        cp_async16(sbase + swz(srow + i * RPP, schunk), pa + (u64)(i * RPP) * PL_K);
#pragma unroll
    for (int i = 0; i < BPASS; ++i)
        cp_async16(sbase + STAGE_A + swz(srow + i * RPP, schunk), pb + (u64)(i * RPP) * PL_K);
}

// ---------------------------------------------------------------- BLAKE3 (un bloque de 64 B)
__device__ __forceinline__ u32 rotr32(u32 x, int n) { return __funnelshift_r(x, x, n); }

#define G(a, b, c, d, x, y)                     \
    a = a + b + x; d = rotr32(d ^ a, 16);       \
    c = c + d;     b = rotr32(b ^ c, 12);       \
    a = a + b + y; d = rotr32(d ^ a, 8);        \
    c = c + d;     b = rotr32(b ^ c, 7);

// Compresión BLAKE3 de un único bloque de 64 B: CHUNK_START|CHUNK_END|ROOT|KEYED_HASH = 1|2|8|16.
// Devuelve las 8 palabras de salida en `out` (hash de 32 B, little-endian).
__device__ __forceinline__ void blake3_keyed64(const u32* key, u32* m, u32* out) {
    u32 v0 = key[0], v1 = key[1], v2 = key[2], v3 = key[3];
    u32 v4 = key[4], v5 = key[5], v6 = key[6], v7 = key[7];
    u32 v8 = 0x6A09E667u, v9 = 0xBB67AE85u, v10 = 0x3C6EF372u, v11 = 0xA54FF53Au;
    u32 v12 = 0, v13 = 0, v14 = 64, v15 = 1 | 2 | 8 | 16;
#pragma unroll
    for (int r = 0; r < 7; ++r) {
        G(v0, v4, v8, v12, m[0], m[1]);
        G(v1, v5, v9, v13, m[2], m[3]);
        G(v2, v6, v10, v14, m[4], m[5]);
        G(v3, v7, v11, v15, m[6], m[7]);
        G(v0, v5, v10, v15, m[8], m[9]);
        G(v1, v6, v11, v12, m[10], m[11]);
        G(v2, v7, v8, v13, m[12], m[13]);
        G(v3, v4, v9, v14, m[14], m[15]);
        if (r < 6) {
            // Permutación de mensaje de BLAKE3: 2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8
            u32 t[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) t[i] = m[i];
            m[0] = t[2]; m[1] = t[6]; m[2] = t[3]; m[3] = t[10];
            m[4] = t[7]; m[5] = t[0]; m[6] = t[4]; m[7] = t[13];
            m[8] = t[1]; m[9] = t[11]; m[10] = t[12]; m[11] = t[5];
            m[12] = t[9]; m[13] = t[14]; m[14] = t[15]; m[15] = t[8];
        }
    }
    out[0] = v0 ^ v8; out[1] = v1 ^ v9; out[2] = v2 ^ v10; out[3] = v3 ^ v11;
    out[4] = v4 ^ v12; out[5] = v5 ^ v13; out[6] = v6 ^ v14; out[7] = v7 ^ v15;
}
#undef G

// Parámetros por vuelta en memoria constante del kernel (los escribe el host o un kernel previo).
struct MineParams {
    u32 key[8];      // a_noise_seed como 8 palabras little-endian
    u32 bound[8];    // cota de share (target × factor), 256 bits little-endian, palabra 7 la más alta
    u32 round_id;    // identificador de la vuelta, se copia al candidato
    u32 pad[7];
};

// Comparación de 256 bits little-endian: hash ≤ bound.
__device__ __forceinline__ bool le256(const u32* h, const u32* b) {
#pragma unroll
    for (int i = 7; i >= 0; --i) {
        if (h[i] != b[i]) return h[i] < b[i];
    }
    return true;
}

// Candidato: vuelta, fila base, columna base y el hash (para depurar).
struct Candidate {
    u32 round_id, row0, col0, flags;
    u32 hash[8];
};

// a: A' (m×k), bt: B'ᵀ (n×k). Grid: (m/BM) × (n/BN) bloques, m-bloque más rápido para reutilizar B en L2.
#if PL_CLUSTER
extern "C" __global__ void __cluster_dims__(PL_CLUSTER, 1, 1) __launch_bounds__(NTHREADS, PL_MINB)
#else
extern "C" __global__ void __launch_bounds__(NTHREADS, PL_MINB)
#endif
pl02_mine(const signed char* __restrict__ a, const signed char* __restrict__ bt, int m, int n,
          const MineParams* __restrict__ params, u32* __restrict__ out, Candidate* __restrict__ cands,
          u32* __restrict__ ncands, u32 max_cands
#if PL_TMA
          , const __grid_constant__ TmaMap map_a, const __grid_constant__ TmaMap map_b
#endif
) {
    extern __shared__ __align__(128) unsigned char smem[];
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;
    const int mblocks = m / BM;
#if PL_GA
    const int ola = blockIdx.x / (PL_GA * PL_GB), en_ola = blockIdx.x % (PL_GA * PL_GB);
    const int bm = (ola % (mblocks / PL_GA)) * PL_GA + en_ola % PL_GA;
    const int bn = (ola / (mblocks / PL_GA)) * PL_GB + en_ola / PL_GA;
#else
    const int bm = blockIdx.x % mblocks, bn = blockIdx.x / mblocks;
#endif
    const int m0 = bm * BM, n0 = bn * BN;

    const signed char* ag = a + (u64)m0 * PL_K;
    const signed char* bg = bt + (u64)n0 * PL_K;
    const u32 sbase = smem_u32(smem);

    int acc[MT][NTL][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NTL; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0;
    u32 jp[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) jp[i] = 0;

    // Direcciones de ldmatrix dentro de una etapa (sin el trozo k, que se añade por paso).
    const int lm = lane >> 3, lr = lane & 7;
    // A: matriz lm → filas (lm&1)*8 + lr de la m-tesela, trozo 2s + (lm>>1).
    const u32 a_row0 = wm * WM + (lm & 1) * 8 + lr;
    const u32 a_chi = lm >> 1;
    // B: matriz lm → n-tesela (lm>>1) del par, trozo 2s + (lm&1).
    const u32 b_row0 = wn * WN + (lm >> 1) * 8 + lr;
    const u32 b_chi = lm & 1;

    constexpr int KT = PL_K / BK;
#ifdef PL_PROBE_REGS
    u32 af[MT][4], bf[NTL][2];
#endif
#if PL_TMA
    // Un mbarrier «lleno» de 8 B por etapa, detrás de las etapas; con PL_TMA_EMPTY, otro «vacío» por
    // etapa detrás, al que llega cada warp cuando ha leído la etapa entera.
    const u32 mbase = sbase + PL_STAGES * STAGE_BYTES;
#if PL_TMA_EMPTY
    const u32 ebase = mbase + PL_STAGES * 8;
#endif
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < PL_STAGES; ++s) {
            mbar_init(mbase + s * 8, 1);
#if PL_TMA_EMPTY == 1
            mbar_init(ebase + s * 8, NTHREADS / 32 * (PL_CLUSTER ? PL_CLUSTER : 1));
#elif PL_TMA_EMPTY == 2
            *(volatile u32*)(smem + PL_STAGES * STAGE_BYTES + PL_STAGES * 8 + s * 8) = 0;
#endif
        }
    }
#if PL_CLUSTER
    cluster_sync();   // los mbarrier de los dos bloques, listos antes de cualquier multicast
#else
    __syncthreads();
#endif
    if (tid == 0) {
        // Con barreras «vacío» todas las etapas empiezan libres: se llenan todas.
#pragma unroll
        for (int s = 0; s < PL_STAGES - (PL_TMA_EMPTY ? 0 : 1); ++s) {
            mbar_expect(mbase + s * 8, STAGE_BYTES);
            tma_tile(sbase + s * STAGE_BYTES, &map_a, TA(s * BK), mbase + s * 8);
            tma_b(sbase + s * STAGE_BYTES + STAGE_A, &map_b, TB(s * BK), mbase + s * 8);
        }
    }
#else
    // Puntero de este hilo a su fila y trozo en k = 0; avanza BK por etapa cargada.
    const u32 srow = (u32)tid / CHUNKS, schunk = (u32)tid % CHUNKS;
    const signed char* pa = ag + (u64)srow * PL_K + schunk * 16;
    const signed char* pb = bg + (u64)srow * PL_K + schunk * 16;
#pragma unroll
    for (int s = 0; s < PL_STAGES - 1; ++s) {
        load_stage(sbase + s * STAGE_BYTES, pa, pb, srow, schunk);
        pa += BK; pb += BK;
        cp_commit();
    }
#endif

#pragma unroll 1
    for (int kt = 0; kt < KT; ++kt) {
#if PL_TMA && PL_TMA_EMPTY
        // Sólo se espera a que llegue la tesela; la etapa se recarga al final de la vuelta.
#ifdef PL_PROBE_NOLOOPLOAD
        if (kt < PL_STAGES)   // sonda: sólo las cargas del prólogo; luego reutiliza las etapas (inválido)
#endif
        mbar_wait(mbase + (kt % PL_STAGES) * 8, (kt / PL_STAGES) & 1);
#elif PL_TMA
        // La tesela de esta vuelta ya está; el __syncthreads garantiza además que nadie
        // sigue leyendo la etapa que vamos a sobrescribir (la de la vuelta anterior).
        mbar_wait(mbase + (kt % PL_STAGES) * 8, (kt / PL_STAGES) & 1);
        __syncthreads();
        {
            int nk = kt + PL_STAGES - 1;
#ifndef PL_PROBE_NOGLOBAL
            if (nk < KT && tid == 0) {
                const u32 sl = (nk % PL_STAGES);
                mbar_expect(mbase + sl * 8, STAGE_BYTES);
                tma_tile(sbase + sl * STAGE_BYTES, &map_a, TA(nk * BK), mbase + sl * 8);
                tma_tile(sbase + sl * STAGE_BYTES + STAGE_A, &map_b, TB(nk * BK), mbase + sl * 8);
            }
#endif
        }
#else
#ifdef PL_PROBE_NOWAIT
        cp_wait<PL_STAGES - 1>();   // sonda: emite las cargas pero no espera (resultado inválido)
#else
        cp_wait<PL_STAGES - 2>();
#endif
        __syncthreads();
        {
            int nk = kt + PL_STAGES - 1;
#ifndef PL_PROBE_NOGLOBAL
            if (nk < KT) load_stage(sbase + (nk % PL_STAGES) * STAGE_BYTES, pa, pb, srow, schunk);
            pa += BK; pb += BK;
#endif
            cp_commit();
        }
#endif
        const u32 st = sbase + (kt % PL_STAGES) * STAGE_BYTES;
#pragma unroll
        for (int s = 0; s < KSTEPS; ++s) {
#ifndef PL_PROBE_REGS
            u32 af[MT][4];
            u32 bf[NTL][2];
#else
            if (kt == 0) {
#endif
#pragma unroll
            for (int i = 0; i < MT; ++i)
                ldsm_x4(st + swz(a_row0 + i * 16, 2 * s + a_chi), af[i][0], af[i][1], af[i][2], af[i][3]);
#pragma unroll
            for (int j = 0; j < NTL; j += 2)
                ldsm_x4(st + STAGE_A + swz(b_row0 + j * 8, 2 * s + b_chi), bf[j][0], bf[j][1], bf[j + 1][0],
                        bf[j + 1][1]);
#ifdef PL_PROBE_REGS
            }
#endif
#if PL_TMA && PL_TMA_EMPTY == 1
            // Los fragmentos del último paso ya están en registros: el warp libera la etapa.
            if (s == KSTEPS - 1) {
                __syncwarp();
                if (lane == 0) {
                    mbar_arrive(ebase + (kt % PL_STAGES) * 8);
#if PL_CLUSTER
                    mbar_arrive_remote(peer_addr(ebase + (kt % PL_STAGES) * 8, cluster_rank() ^ 1u));
#endif
                }
#if PL_ISSUE_EARLY
                // El productor pide ya la tesela kt + STAGES, antes de su última tanda de mma.
                if (tid == 0 && kt + PL_STAGES < KT) {
                    const u32 sl = kt % PL_STAGES;
                    const int nk = kt + PL_STAGES;
                    mbar_wait(ebase + sl * 8, (kt / PL_STAGES) & 1);
                    mbar_expect(mbase + sl * 8, STAGE_BYTES);
                    tma_tile(sbase + sl * STAGE_BYTES, &map_a, TA(nk * BK), mbase + sl * 8);
                    tma_tile(sbase + sl * STAGE_BYTES + STAGE_A, &map_b, TB(nk * BK), mbase + sl * 8);
                }
#endif
            }
#elif PL_TMA && PL_TMA_EMPTY == 2
            // Igual, pero con un contador: el warp que llega el último pide ya la tesela kt + STAGES
            // en esta etapa. Ningún warp espera a otro; sólo a que lleguen los datos.
            if (s == KSTEPS - 1) {
                __syncwarp();
                if (lane == 0) {
                    const u32 sl = kt % PL_STAGES;
                    const int nk = kt + PL_STAGES;
                    u32 old;
                    __threadfence_block();
                    asm volatile("atom.shared.add.u32 %0, [%1], 1;\n" : "=r"(old) : "r"(ebase + sl * 8) : "memory");
                    if (old == (u32)(NTHREADS / 32) * (u32)(kt / PL_STAGES + 1) - 1u && nk < KT) {
                        // Las lecturas de los demás warps (proxy genérico) antes de la escritura de TMA.
                        __threadfence_block();
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                        mbar_expect(mbase + sl * 8, STAGE_BYTES);
                        tma_tile(sbase + sl * STAGE_BYTES, &map_a, TA(nk * BK), mbase + sl * 8);
                        tma_tile(sbase + sl * STAGE_BYTES + STAGE_A, &map_b, TB(nk * BK), mbase + sl * 8);
                    }
                }
            }
#endif
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NTL; ++j) mma_s8(acc[i][j], af[i], bf[j][0], bf[j][1]);
        }
#ifndef PL_PROBE_NOFOLD
        if ((kt & (KT_PER_TRAMO - 1)) == KT_PER_TRAMO - 1) {
            // Fin de tramo de 128: XOR de las 128 sumas acumuladas y actualización del jackpot.
            // En árbol, no en cadena: escrito como `x ^= …` el compilador encadena ~64 LOP3
            // dependientes (≈256 ciclos de latencia en serie) justo entre dos grupos de mma.
#if PL_FOLD_TREE
            u32 p[MT * NTL];
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NTL; ++j)
                    p[i * NTL + j] = (u32)acc[i][j][0] ^ (u32)acc[i][j][1] ^ (u32)acc[i][j][2] ^
                                     (u32)acc[i][j][3];
#pragma unroll
            for (int cnt = MT * NTL; cnt > 1; cnt = (cnt + 1) / 2) {   // también si MT·NTL no es potencia de dos
#pragma unroll
                for (int t2 = 0; t2 < cnt / 2; ++t2) p[t2] = p[2 * t2] ^ p[2 * t2 + 1];
                if (cnt & 1) p[cnt / 2] = p[cnt - 1];
            }
            const u32 x = p[0];
#else
            u32 x = 0;
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NTL; ++j)
                    x ^= (u32)acc[i][j][0] ^ (u32)acc[i][j][1] ^ (u32)acc[i][j][2] ^ (u32)acc[i][j][3];
#endif
            u32 t = __funnelshift_l(jp[0], jp[0], 13) ^ x;
#pragma unroll
            for (int i = 0; i < 15; ++i) jp[i] = jp[i + 1];
            jp[15] = t;
        }
#endif
#if PL_TMA && PL_TMA_EMPTY == 1
        // Productor: cuando los warps han soltado la etapa, se pide en ella la tesela kt + STAGES.
        // Sólo espera el warp 0; los demás siguen con la etapa siguiente, que ya está pedida.
#if defined(PL_PROBE_NOLOOPLOAD) || PL_ISSUE_EARLY
        if (false) {
#else
        if (tid == 0 && kt + PL_STAGES < KT) {
#endif
            const u32 sl = kt % PL_STAGES;
            const int nk = kt + PL_STAGES;
#if PL_CLUSTER
            mbar_wait_cluster(ebase + sl * 8, (kt / PL_STAGES) & 1);
#else
            mbar_wait(ebase + sl * 8, (kt / PL_STAGES) & 1);
#endif
#if defined(PL_PROBE_LOADONLY) && PL_PROBE_LOADONLY == 1   // sonda: sólo A en el bucle (inválido)
            mbar_expect(mbase + sl * 8, STAGE_A);
            tma_tile(sbase + sl * STAGE_BYTES, &map_a, TA(nk * BK), mbase + sl * 8);
#elif defined(PL_PROBE_LOADONLY) && PL_PROBE_LOADONLY == 2 // sonda: sólo B en el bucle (inválido)
            mbar_expect(mbase + sl * 8, STAGE_B);
            tma_tile(sbase + sl * STAGE_BYTES + STAGE_A, &map_b, TB(nk * BK), mbase + sl * 8);
#else
            mbar_expect(mbase + sl * 8, STAGE_BYTES);
            tma_tile(sbase + sl * STAGE_BYTES, &map_a, TA(nk * BK), mbase + sl * 8);
            tma_b(sbase + sl * STAGE_BYTES + STAGE_A, &map_b, TB(nk * BK), mbase + sl * 8);
#endif
        }
#endif
    }
#if !PL_TMA
    cp_wait<0>();
#endif
#if PL_CLUSTER
    cluster_sync();   // el otro bloque aún puede llegar a nuestras barreras «vacío»
#endif

    const int g = lane >> 2, tq = lane & 3;
    const u32 row0 = m0 + wm * WM + g;          // fila base de la tesela de hash
    const u32 col0 = n0 + wn * WN + 2 * tq;     // columna base
    const u32 rt = (u32)(bm * (BM / WM) + wm) * 8 + g;   // tesela de hash: 8 por banda de WM filas
    const u32 ct = (u32)(bn * (BN / WN) + wn) * 4 + tq;  // 4 por banda de WN columnas
    const u64 tile = (u64)rt * (u32)(n / WN * 4) + ct;
#if PL_MODE == 0
#pragma unroll
    for (int i = 0; i < 16; ++i) out[tile * 16 + i] = jp[i];
#elif PL_MODE == 1
    u32 x = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) x ^= jp[i];
    out[tile] = x;
#else
    u32 key[8], bound[8], h[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { key[i] = params->key[i]; bound[i] = params->bound[i]; }
    blake3_keyed64(key, jp, h);
    if (out) {
#pragma unroll
        for (int i = 0; i < 8; ++i) out[tile * 8 + i] = h[i];
    }
    if (le256(h, bound)) {
        u32 slot = atomicAdd(ncands, 1u);
        if (slot < max_cands) {
            Candidate c;
            c.round_id = params->round_id; c.row0 = row0; c.col0 = col0; c.flags = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) c.hash[i] = h[i];
            cands[slot] = c;
        }
    }
#endif
}
