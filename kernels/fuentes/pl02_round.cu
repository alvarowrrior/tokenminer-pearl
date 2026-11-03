// Vuelta de Pearl en GPU (H3): matrices desde semilla, raíz Merkle, salado V3, semillas y ruido.
// Código propio de TokenMiner. La especificación sigue a pearl-research-labs/pearl (licencia ISC):
// raíz = BLAKE3 con clave job_key del relleno a 1024 B (ffi/mine.rs, compute_commitment_hash_with_offsets),
// salado y semillas (api/seed.rs, PublicProofParams::commitment_hash), ruido (circuit/pearl_noise.rs).
//
// Matrices: receta con semilla propia por matriz. La palabra i de la matriz
// por filas es mezcla(i + semilla[i mod 8]); sus cuatro bytes, (b & 127) − 64, en [−64, 63].
// A nunca se guarda: el Merkle y A' la regeneran en registros. Por vuelta sólo se escribe A'.
// Con PL_CTE (forma 2) A y Bᵀ valen 64 en todos los bytes, salvo la última porción de A, que lleva la
// receta y cambia en cada vuelta: basta para renovar la raíz y las semillas, y A' y B' son 64 + ruido, que
// gasta menos en los tensor cores. Esa porción cae en una fila que el producto no usa.
//
// Orden por trabajo: chunks(B) → parents… → tree_final(B) → tables(B) → noised(B).
// Orden por vuelta:  begin → chunks(A) → tree_final(A) → tables(A) → noised(A) → producto (6 kernels).
// Con PL_CTE sólo cambia la última porción de A: por trabajo, chunks y tree_final guardan los hermanos de su
// camino hasta la raíz, que no cambian; por vuelta, ronda → tables(A) → noised(A) → producto (4 kernels).

#ifndef PL_K
#define PL_K 4096
#endif
#ifndef PL_CTE
#define PL_CTE 0
#endif
#define PL_RANK 128   // rango del ruido; las cuentas de abajo (128 B por fila de U, máscara 127) lo fijan

typedef unsigned int u32;
typedef unsigned char u8;

static_assert(PL_K % 1024 == 0, "cada fila de A ocupa porciones de 1024 B enteras");

// Lo que el host fija por trabajo.
struct Job {
    u32 job_key[8];   // BLAKE3(header ‖ configuración)
    u32 seed_b[8];    // semilla de la receta de Bᵀ
    u32 seed_a0[8];   // semilla de la receta de A en la vuelta 0; la vuelta r suma r a la palabra 0
    u32 bound[8];     // cota de share (target × factor), 256 bits little-endian
    u32 b_seed[8];    // b_noise_seed: lo escribe pl02_seed(B)
    u32 m, n, pad[6];
};
// Estado de la vuelta en curso (lo avanza pl02_round_begin).
struct Round {
    u32 seed_a[8];
    u32 round_id;
    u32 pad[7];
};
// La misma estructura que lee el producto (pl02_core.cu).
struct MineParams {
    u32 key[8];
    u32 bound[8];
    u32 round_id;
    u32 pad[7];
};

// ---------------------------------------------------------------- BLAKE3
#define F_START 1u
#define F_END 2u
#define F_PARENT 4u
#define F_ROOT 8u
#define F_KEYED 16u

__device__ __constant__ u32 IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                                     0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
// blake3("pearl/cert-v3/noise-seed/A") y …/B (SEED_SALT_A/B de api/seed.rs), en palabras little-endian.
__device__ __constant__ u32 SALT[2][8] = {
    {0x6C404982u, 0x1615EDA0u, 0x92F61696u, 0xF876F0FCu, 0x2ADBDB92u, 0x52B82370u, 0x1977D4F0u, 0x7B0190C3u},
    {0x32063011u, 0xCA0163ECu, 0x71AFE22Bu, 0x4F4D3F8Bu, 0x39C6E91Au, 0x04CCE888u, 0x1D304448u, 0xA99AB871u}};

__device__ __forceinline__ u32 rotr32(u32 x, int n) { return __funnelshift_r(x, x, n); }

#define G(a, b, c, d, x, y)                     \
    a = a + b + x; d = rotr32(d ^ a, 16);       \
    c = c + d;     b = rotr32(b ^ c, 12);       \
    a = a + b + y; d = rotr32(d ^ a, 8);        \
    c = c + d;     b = rotr32(b ^ c, 7);

// Compresión de un bloque de 64 B; devuelve el valor encadenado (8 palabras).
__device__ __forceinline__ void compress(const u32* cv, const u32* msg, u32 counter, u32 flags, u32* out) {
    u32 m[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = msg[i];
    u32 v0 = cv[0], v1 = cv[1], v2 = cv[2], v3 = cv[3], v4 = cv[4], v5 = cv[5], v6 = cv[6], v7 = cv[7];
    u32 v8 = IV[0], v9 = IV[1], v10 = IV[2], v11 = IV[3];
    u32 v12 = counter, v13 = 0, v14 = 64, v15 = flags;
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

// ---------------------------------------------------------------- receta de matrices
__device__ __forceinline__ u32 gen_word(u32 i, const u32* seed) {
    u32 h = i + seed[i & 7];
    h = (h ^ (h >> 16)) * 0x7feb352du;
    h = (h ^ (h >> 15)) * 0x846ca68bu;
    h ^= h >> 16;
    // Cada byte x = b & 127 pasa a x − 64 (mod 256) sin acarreo entre bytes: si x ≥ 64 es x ^ 0x40;
    // si no, x + 0xC0 = x ^ 0xC0. Es decir, x ^ 0x40 y además el bit 7 si el bit 6 estaba a 0.
    const u32 x = h & 0x7F7F7F7Fu;
    return x ^ 0x40404040u ^ ((~x & 0x40404040u) << 1);
}

#define WORD_CTE 0x40404040u

// Palabra i de la matriz, que cae en la porción c; `ultima` es la porción con receta (forma 2).
__device__ __forceinline__ u32 mat_word(u32 i, const u32* seed, u32 c, u32 ultima) {
#if PL_CTE
    return c == ultima ? gen_word(i, seed) : WORD_CTE;
#else
    return gen_word(i, seed);
#endif
}

// ---------------------------------------------------------------- kernels
// Vuelta nueva: round_id + 1 y semilla de A de esa vuelta (palabra 0 + round_id). Un hilo.
extern "C" __global__ void pl02_round_begin(const Job* __restrict__ job, Round* __restrict__ rd) {
    const u32 r = rd->round_id + 1;
    rd->round_id = r;
#pragma unroll
    for (int i = 0; i < 8; ++i) rd->seed_a[i] = job->seed_a0[i] + (i == 0 ? r : 0u);
}

// Padre de dos valores encadenados (16 palabras seguidas): BLAKE3 con clave, PARENT y, si toca, ROOT.
__device__ __forceinline__ void parent(const u32* key, const u32* lr, u32 flags, u32* out) {
    compress(key, lr, 0, F_PARENT | F_KEYED | flags, out);
}

#define TREE_T 128   // porciones por bloque en pl02_chunks: el bloque devuelve la raíz de su subárbol

// Hojas del Merkle y los 7 primeros niveles: cada hilo hace una porción de 1024 B (16 bloques, contador
// = índice de porción) y el bloque reduce sus 128 valores en compartida. El árbol es perfecto (número de
// porciones potencia de dos), así que los subárboles de 128 porciones consecutivas son exactos.
// Nunca lleva ROOT: el host exige al menos 256 porciones y la raíz la hace pl02_tree_final.
extern "C" __global__ void __launch_bounds__(TREE_T) pl02_chunks(const u32* __restrict__ seed,
                                                                 const u32* __restrict__ key,
                                                                 u32* __restrict__ cv_out, u32 ultima,
                                                                 u32* __restrict__ espina) {
    __shared__ u32 cvs[TREE_T * 8];
    const u32 c = blockIdx.x * TREE_T + threadIdx.x;
    u32 s[8], k[8], cv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { s[i] = seed[i]; k[i] = key[i]; cv[i] = k[i]; }
#pragma unroll 1
    for (u32 b = 0; b < 16; ++b) {
        u32 msg[16];
#pragma unroll
        for (u32 i = 0; i < 16; ++i) msg[i] = mat_word(c * 256 + b * 16 + i, s, c, ultima);
        compress(cv, msg, c, F_KEYED | (b == 0 ? F_START : 0u) | (b == 15 ? F_END : 0u), cv);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) cvs[threadIdx.x * 8 + i] = cv[i];
    __syncthreads();
    u32 nivel = 0;
#pragma unroll 1
    for (u32 cnt = TREE_T / 2; cnt >= 1; cnt >>= 1, ++nivel) {
        u32 lr[16], out[8];
        // Hermano izquierdo del camino de la última porción en este nivel.
        if (espina && blockIdx.x == gridDim.x - 1 && threadIdx.x == 2 * cnt - 2) {
#pragma unroll
            for (int i = 0; i < 8; ++i) espina[nivel * 8 + i] = cvs[threadIdx.x * 8 + i];
        }
        if (threadIdx.x < cnt) {
#pragma unroll
            for (int i = 0; i < 16; ++i) lr[i] = cvs[threadIdx.x * 16 + i];
            parent(k, lr, 0, out);
        }
        __syncthreads();
        if (threadIdx.x < cnt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) cvs[threadIdx.x * 8 + i] = out[i];
        }
        __syncthreads();
    }
    if (threadIdx.x < 8) cv_out[blockIdx.x * 8 + threadIdx.x] = cvs[threadIdx.x];
}

// Un nivel de padres fuera de bloque: nout nodos a partir de 2·nout hijos. Sólo hace falta para
// matrices de más de 1024·128 porciones (B por trabajo); nunca es el último nivel.
extern "C" __global__ void pl02_parents(const u32* __restrict__ in, u32* __restrict__ out, u32 nout,
                                        const u32* __restrict__ key) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nout) return;
    u32 lr[16], k[8], cv[8];
#pragma unroll
    for (int j = 0; j < 16; ++j) lr[j] = in[i * 16 + j];
#pragma unroll
    for (int j = 0; j < 8; ++j) k[j] = key[j];
    parent(k, lr, 0, cv);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[i * 8 + j] = cv[j];
}

// Raíz del árbol (los niveles que quedan, el último con ROOT), salado V3 y semilla de ruido.
// Un bloque de nin/2 hilos, nin potencia de dos entre 2 y 1024.
// side 1 (B): b_seed = BLAKE3(job_key ‖ B_salada) → job.
// side 0 (A): a_seed = BLAKE3(b_seed ‖ A_salada) → parámetros del producto, con la cota y la vuelta.
extern "C" __global__ void __launch_bounds__(512) pl02_tree_final(const u32* __restrict__ cv_in, u32 nin,
                                                                  Job* __restrict__ job, const Round* __restrict__ rd,
                                                                  int side, MineParams* __restrict__ mp,
                                                                  u32* __restrict__ root_out, u32* __restrict__ espina) {
    __shared__ u32 cvs[1024 * 8];
    __shared__ u32 k[8];
    const u32 t = threadIdx.x;
    for (u32 i = t; i < 8; i += blockDim.x) k[i] = job->job_key[i];   // el bloque puede tener menos de 8 hilos
    for (u32 i = t; i < nin * 8; i += blockDim.x) cvs[i] = cv_in[i];
    __syncthreads();
    u32 kk[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) kk[i] = k[i];
    u32 nivel = 7;   // los 7 primeros los hace pl02_chunks
#pragma unroll 1
    for (u32 cnt = nin / 2; cnt >= 1; cnt >>= 1, ++nivel) {
        u32 lr[16], out[8];
        if (espina && t == 0) {   // el bloque tiene cnt hilos: el hilo 2·cnt − 2 no existe
#pragma unroll
            for (int i = 0; i < 8; ++i) espina[nivel * 8 + i] = cvs[(2 * cnt - 2) * 8 + i];
        }
        if (t < cnt) {
#pragma unroll
            for (int i = 0; i < 16; ++i) lr[i] = cvs[t * 16 + i];
            parent(kk, lr, cnt == 1 ? F_ROOT : 0u, out);
        }
        __syncthreads();
        if (t < cnt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) cvs[t * 8 + i] = out[i];
        }
        __syncthreads();
    }
    if (t != 0) return;
    u32 msg[16], salted[8], seed[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { msg[i] = cvs[i]; root_out[i] = cvs[i]; }
    msg[8] = side ? job->n : job->m;
#pragma unroll
    for (int i = 9; i < 16; ++i) msg[i] = 0;
    compress(SALT[side], msg, 0, F_START | F_END | F_ROOT | F_KEYED, salted);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        msg[i] = side ? job->job_key[i] : job->b_seed[i];
        msg[8 + i] = salted[i];
    }
    compress(IV, msg, 0, F_START | F_END | F_ROOT, seed);
    if (side) {
#pragma unroll
        for (int i = 0; i < 8; ++i) job->b_seed[i] = seed[i];
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) { mp->key[i] = seed[i]; mp->bound[i] = job->bound[i]; }
        mp->round_id = rd->round_id;
    }
}

// Vuelta nueva con A constante (forma 2), en un hilo: round_id + 1, semilla de A, su última porción, el camino
// hasta la raíz con los hermanos de `espina`, salado V3 y a_seed. `niveles` = log2(porciones).
extern "C" __global__ void pl02_ronda(const Job* __restrict__ job, Round* __restrict__ rd, const u32* __restrict__ espina,
                                      u32 porciones, u32 niveles, MineParams* __restrict__ mp, u32* __restrict__ root_out) {
    const u32 r = rd->round_id + 1;
    rd->round_id = r;
    u32 s[8], k[8], cv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        s[i] = job->seed_a0[i] + (i == 0 ? r : 0u);
        rd->seed_a[i] = s[i];
        k[i] = job->job_key[i];
        cv[i] = k[i];
    }
    const u32 c = porciones - 1;
#pragma unroll 1
    for (u32 b = 0; b < 16; ++b) {
        u32 msg[16];
#pragma unroll
        for (u32 i = 0; i < 16; ++i) msg[i] = gen_word(c * 256 + b * 16 + i, s);
        compress(cv, msg, c, F_KEYED | (b == 0 ? F_START : 0u) | (b == 15 ? F_END : 0u), cv);
    }
#pragma unroll 1
    for (u32 v = 0; v < niveles; ++v) {
        u32 lr[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) { lr[i] = espina[v * 8 + i]; lr[8 + i] = cv[i]; }
        parent(k, lr, v + 1 == niveles ? F_ROOT : 0u, cv);
    }
    u32 msg[16], salted[8], seed[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { msg[i] = cv[i]; root_out[i] = cv[i]; }
    msg[8] = job->m;
#pragma unroll
    for (int i = 9; i < 16; ++i) msg[i] = 0;
    compress(SALT[0], msg, 0, F_START | F_END | F_ROOT | F_KEYED, salted);
#pragma unroll
    for (int i = 0; i < 8; ++i) { msg[i] = job->b_seed[i]; msg[8 + i] = salted[i]; }
    compress(IV, msg, 0, F_START | F_END | F_ROOT, seed);
#pragma unroll
    for (int i = 0; i < 8; ++i) { mp->key[i] = seed[i]; mp->bound[i] = job->bound[i]; }
    mp->round_id = r;
}

// Mensaje de get_random_hash: (1 + índice) en la ranura `slot`, etiqueta "A_tensor"/"B_tensor" detrás.
__device__ __forceinline__ void noise_msg(u32* msg, u32 index, int slot, int side) {
#pragma unroll
    for (int i = 0; i < 16; ++i) msg[i] = 0;
    msg[slot] = index + 1;
    msg[8] = 0x65745F41u + (u32)side;   // "A_te" / "B_te"
    msg[9] = 0x726F736Eu;               // "nsor"
}

// Tablas del ruido de un lado, en un solo kernel: los hilos [0, rows·4) calculan la matriz uniforme
// (un hash de 32 B cada uno; la fila r ocupa los bytes r·128 … r·128 + 127, (b & 63) − 32) y los
// hilos [rows·4, rows·4 + k/8) la de permutación (8 parejas (first, second) por hash).
extern "C" __global__ void pl02_tables(const u32* __restrict__ key, int side, u32 rows, signed char* __restrict__ u,
                                       u8* __restrict__ pairs) {
    const u32 g = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 nu = rows * (PL_RANK / 32);
    if (g >= nu + PL_K / 8) return;
    const bool uni = g < nu;
    const u32 i = uni ? g : g - nu;
    u32 msg[16], k[8], h[8];
    noise_msg(msg, i, uni ? 0 : 1, side);
#pragma unroll
    for (int j = 0; j < 8; ++j) k[j] = key[j];
    compress(k, msg, 0, F_START | F_END | F_ROOT | F_KEYED, h);
    if (uni) {
        // Cada byte y = b & 63 pasa a y − 32 (mod 256): y ^ 0x20, y además 0xC0 si el bit 5 estaba a 0.
        uint4* out = (uint4*)(u + i * 32);
        u32 o[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const u32 y = h[j] & 0x3F3F3F3Fu, z = ~y & 0x20202020u;
            o[j] = y ^ 0x20202020u ^ (z << 1) ^ (z << 2);
        }
        out[0] = make_uint4(o[0], o[1], o[2], o[3]);
        out[1] = make_uint4(o[4], o[5], o[6], o[7]);
    } else {
        u32 o[4];
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            const u32 f0 = h[j] & (PL_RANK - 1), f1 = h[j + 1] & (PL_RANK - 1);
            const u32 s0 = f0 ^ (1u + __umulhi(PL_RANK - 1, h[j]));
            const u32 s1 = f1 ^ (1u + __umulhi(PL_RANK - 1, h[j + 1]));
            o[j / 2] = f0 | (s0 << 8) | (f1 << 16) | (s1 << 24);
        }
        ((uint4*)pairs)[i] = make_uint4(o[0], o[1], o[2], o[3]);
    }
}

#define NOISE_T (PL_K / 16)   // hilos por fila en pl02_noised: 16 bytes de k cada uno

// Matriz con ruido, un bloque por fila: cada hilo escribe 16 bytes, receta + U[first[l]] − U[second[l]],
// con la fila de U (128 B) en compartida. tile_rows = 0: por filas. Si no, por teselas de carga de
// tile_rows filas × 128 B, en orden (bloque de filas, etapa, fila), como PL_TILED del producto.
extern "C" __global__ void __launch_bounds__(NOISE_T) pl02_noised(const u32* __restrict__ seed,
                                                                  const signed char* __restrict__ uglob,
                                                                  const u8* __restrict__ pairs,
                                                                  uint4* __restrict__ out, u32 tile_rows) {
    __shared__ __align__(16) signed char u[PL_RANK];
    const u32 row = blockIdx.x, t = threadIdx.x;
    if (t < PL_RANK / 16) ((uint4*)u)[t] = ((const uint4*)(uglob + row * PL_RANK))[t];
    u32 s[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) s[j] = seed[j];
    const u32 w0 = (row * (PL_K / 4)) + t * 4;   // primera palabra de la receta de este hilo
    const uint4 pa = ((const uint4*)pairs)[t * 2];      // parejas de los 8 primeros k
    const uint4 pb = ((const uint4*)pairs)[t * 2 + 1];  // y de los 8 siguientes
    const u32 pw[8] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
    __syncthreads();
    u32 o[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
#if PL_CTE
        const u32 w = WORD_CTE;
#else
        const u32 w = gen_word(w0 + q, s);
#endif
        const u32 p01 = pw[q * 2], p23 = pw[q * 2 + 1];   // (first, second) de 4 k: f0 s0 f1 s1 | f2 s2 f3 s3
        u32 v = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const u32 pj = (j < 2 ? p01 : p23) >> ((j & 1) * 16);
            const int e = u[pj & 0xFF] - u[(pj >> 8) & 0xFF];
            v |= ((u32)((int)(signed char)(w >> (j * 8)) + e) & 255u) << (j * 8);
        }
        o[q] = v;
    }
    u32 at = row * NOISE_T + t;
    if (tile_rows) {
        const u32 stage = t / 8, byte16 = t % 8;   // 128 B de k = 8 trozos de 16 B
        at = ((row / tile_rows * (PL_K / 128) + stage) * tile_rows + row % tile_rows) * 8 + byte16;
    }
    out[at] = make_uint4(o[0], o[1], o[2], o[3]);
}

// Contador de candidatos a cero dentro del stream (antes de cada lote de vueltas). Un hilo.
extern "C" __global__ void pl02_reset(u32* __restrict__ counter) { *counter = 0; }
