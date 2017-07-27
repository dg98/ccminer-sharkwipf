/**
* Skunk Algo for Signatum
* (skein, cube, fugue, gost streebog)
*
* tpruvot@github 06 2017 - GPLv3
*/
extern "C" {
#include "sph/sph_skein.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_fugue.h"
#include "sph/sph_streebog.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_vectors.h"

#define h0 h[0]
#define h1 h[1]
#define h2 h[2]
#define h3 h[3]
#define h4 h[4]
#define h5 h[5]
#define h6 h[6]
#define h7 h[7]
#define h8 h[8]
#include "skein/skein_header.h"

extern void skein512_cpu_setBlock_80(void *pdata);
extern void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);

extern void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

extern void x13_fugue512_cpu_init(int thr_id, uint32_t threads);
extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void x13_fugue512_cpu_free(int thr_id);

extern void streebog_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t* d_resNonce);
extern void streebog_set_target(const uint32_t* ptarget);

#include <stdio.h>
#include <memory.h>

#define NBN 2
static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t	*h_resNonce[MAX_GPUS];

// CPU Hash
extern "C" void skunk_hash(void *output, const void *input)
{
	unsigned char _ALIGN(128) hash[128] = { 0 };

	sph_skein512_context ctx_skein;
	sph_cubehash512_context ctx_cubehash;
	sph_fugue512_context ctx_fugue;
	sph_gost512_context ctx_gost;

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, (void*)hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, (const void*)hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*)hash);

	sph_fugue512_init(&ctx_fugue);
	sph_fugue512(&ctx_fugue, (const void*)hash, 64);
	sph_fugue512_close(&ctx_fugue, (void*)hash);

	sph_gost512_init(&ctx_gost);
	sph_gost512(&ctx_gost, (const void*)hash, 64);
	sph_gost512_close(&ctx_gost, (void*)hash);

	memcpy(output, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

#define TPB52 512
#define TPB50 512

static __constant__ uint2 c_buffer[120]; // padded message (80 bytes + 72 bytes midstate + align)

__constant__ const uint2 buffer[152] = {
	{ 0x749C51CE,0x4903ADFF },{ 0x9746DF03,0x0D95DE39 },{ 0x27C79BCE,0x8FD19341 },{ 0xFF352CB1,0x9A255629 },{ 0xDF6CA7B0,0x5DB62599 },{ 0xA9D5C434,0xEABE394C },{ 0x1A75B523,0x891112C7 },{ 0x660FCC33,0xAE18A40B },
	{ 0x9746DF03,0x0D95DE39 },{ 0x27C79BCE,0x8FD19341 },{ 0xFF352CB1,0x9A255629 },{ 0xDF6CA7B0,0x5DB62599 },{ 0xA9D5C3F4,0xEABE394C },{ 0x1A75B523,0x891112C7 },{ 0x660FCC73,0x9E18A40B },{ 0x98173EC5,0xCAB2076D },
	{ 0x27C79BCE,0x8FD19341 },{ 0xFF352CB1,0x9A255629 },{ 0xDF6CA7B0,0x5DB62599 },{ 0xA9D5C3F4,0xEABE394C },{ 0x1A75B523,0x991112C7 },{ 0x660FCC73,0x9E18A40B },{ 0x98173F04,0xCAB2076D },{ 0x749C51D0,0x4903ADFF },
	{ 0xFF352CB1,0x9A255629 },{ 0xDF6CA7B0,0x5DB62599 },{ 0xA9D5C3F4,0xEABE394C },{ 0x1A75B523,0x991112C7 },{ 0x660FCC33,0xAE18A40B },{ 0x98173F04,0xCAB2076D },{ 0x749C51CE,0x3903ADFF },{ 0x9746DF06,0x0D95DE39 },
	{ 0xDF6CA7B0,0x5DB62599 },{ 0xA9D5C3F4,0xEABE394C },{ 0x1A75B523,0x991112C7 },{ 0x660FCC33,0xAE18A40B },{ 0x98173EC4,0xCAB2076D },{ 0x749C51CE,0x3903ADFF },{ 0x9746DF43,0xFD95DE39 },{ 0x27C79BD2,0x8FD19341 },
	{ 0xA9D5C3F4,0xEABE394C },{ 0x1A75B523,0x991112C7 },{ 0x660FCC33,0xAE18A40B },{ 0x98173EC4,0xCAB2076D },{ 0x749C51CE,0x4903ADFF },{ 0x9746DF43,0xFD95DE39 },{ 0x27C79C0E,0x8FD19341 },{ 0xFF352CB6,0x9A255629 },
	{ 0x1A75B523,0x991112C7 },{ 0x660FCC33,0xAE18A40B },{ 0x98173EC4,0xCAB2076D },{ 0x749C51CE,0x4903ADFF },{ 0x9746DF03,0x0D95DE39 },{ 0x27C79C0E,0x8FD19341 },{ 0xFF352CB1,0x8A255629 },{ 0xDF6CA7B6,0x5DB62599 },
	{ 0x660FCC33,0xAE18A40B },{ 0x98173EC4,0xCAB2076D },{ 0x749C51CE,0x4903ADFF },{ 0x9746DF03,0x0D95DE39 },{ 0x27C79BCE,0x8FD19341 },{ 0xFF352CB1,0x8A255629 },{ 0xDF6CA7F0,0x4DB62599 },{ 0xA9D5C3FB,0xEABE394C },
	{ 0x98173EC4,0xCAB2076D },{ 0x749C51CE,0x4903ADFF },{ 0x9746DF03,0x0D95DE39 },{ 0x27C79BCE,0x8FD19341 },{ 0xFF352CB1,0x9A255629 },{ 0xDF6CA7F0,0x4DB62599 },{ 0xA9D5C434,0xEABE394C },{ 0x1A75B52B,0x991112C7 },
	{ 0x749C51CE,0x4903ADFF },{ 0x9746DF03,0x0D95DE39 },{ 0x27C79BCE,0x8FD19341 },{ 0xFF352CB1,0x9A255629 },{ 0xDF6CA7B0,0x5DB62599 },{ 0xA9D5C434,0xEABE394C },{ 0x1A75B523,0x891112C7 },{ 0x660FCC3C,0xAE18A40B },
	{ 0x9746DF03,0x0D95DE39 },{ 0x27C79BCE,0x8FD19341 },{ 0xFF352CB1,0x9A255629 },{ 0xDF6CA7B0,0x5DB62599 },{ 0xA9D5C3F4,0xEABE394C },{ 0x1A75B523,0x891112C7 },{ 0x660FCC73,0x9E18A40B },{ 0x98173ece,0xcab2076d },
	{ 0x27C79BCE,0x8FD19341 },{ 0xFF352CB1,0x9A255629 },{ 0xDF6CA7B0,0x5DB62599 },{ 0xA9D5C3F4,0xEABE394C },{ 0x1A75B523,0x991112C7 },{ 0x660FCC73,0x9E18A40B },{ 0x98173F04,0xCAB2076D },{ 0x749C51D9,0x4903ADFF },
	{ 0xFF352CB1,0x9A255629 },{ 0xDF6CA7B0,0x5DB62599 },{ 0xA9D5C3F4,0xEABE394C },{ 0x1A75B523,0x991112C7 },{ 0x660FCC33,0xAE18A40B },{ 0x98173F04,0xCAB2076D },{ 0x749C51CE,0x3903ADFF },{ 0x9746DF0F,0x0D95DE39 },
	{ 0xDF6CA7B0,0x5DB62599 },{ 0xA9D5C3F4,0xEABE394C },{ 0x1A75B523,0x991112C7 },{ 0x660FCC33,0xAE18A40B },{ 0x98173EC4,0xCAB2076D },{ 0x749C51CE,0x3903ADFF },{ 0x9746DF43,0xFD95DE39 },{ 0x27C79BDB,0x8FD19341 },
	{ 0xA9D5C3F4,0xEABE394C },{ 0x1A75B523,0x991112C7 },{ 0x660FCC33,0xAE18A40B },{ 0x98173EC4,0xCAB2076D },{ 0x749C51CE,0x4903ADFF },{ 0x9746DF43,0xFD95DE39 },{ 0x27C79C0E,0x8FD19341 },{ 0xFF352CBF,0x9A255629 },
	{ 0x1A75B523,0x991112C7 },{ 0x660FCC33,0xAE18A40B },{ 0x98173EC4,0xCAB2076D },{ 0x749C51CE,0x4903ADFF },{ 0x9746DF03,0x0D95DE39 },{ 0x27C79C0E,0x8FD19341 },{ 0xFF352CB1,0x8A255629 },{ 0xDF6CA7BF,0x5DB62599 },
	{ 0x660FCC33,0xAE18A40B },{ 0x98173ec4,0xcab2076d },{ 0x749C51CE,0x4903ADFF },{ 0x9746DF03,0x0D95DE39 },{ 0x27C79BCE,0x8FD19341 },{ 0xFF352CB1,0x8A255629 },{ 0xDF6CA7F0,0x4DB62599 },{ 0xA9D5C404,0xEABE394C },
	{ 0x98173ec4,0xcab2076d },{ 0x749C51CE,0x4903ADFF },{ 0x9746DF03,0x0D95DE39 },{ 0x27C79BCE,0x8FD19341 },{ 0xFF352CB1,0x9A255629 },{ 0xDF6CA7F0,0x4DB62599 },{ 0xA9D5C434,0xEABE394C },{ 0x1A75B534,0x991112C7 },
	{ 0x749C51CE,0x4903ADFF },{ 0x9746DF03,0x0D95DE39 },{ 0x27C79BCE,0x8FD19341 },{ 0xFF352CB1,0x9A255629 },{ 0xDF6CA7B0,0x5DB62599 },{ 0xA9D5C434,0xEABE394C },{ 0x1A75B523,0x891112C7 },{ 0x660FCC45,0xAE18A40B }
};

#define SWAP(a,b) { uint32_t u = a; a = b; b = u; }
__device__ __forceinline__
static void rrounds(uint32_t *x) {
#pragma unroll 2
	for (int r = 0; r < 16; r++) {
		/* "add x_0jklm into x_1jklmn modulo 2^32 rotate x_0jklm upwards by 7 bits" */
		x[16] = x[16] + x[0]; x[0] = ROTL32(x[0], 7); x[17] = x[17] + x[1]; x[1] = ROTL32(x[1], 7);
		x[18] = x[18] + x[2]; x[2] = ROTL32(x[2], 7); x[19] = x[19] + x[3]; x[3] = ROTL32(x[3], 7);
		x[20] = x[20] + x[4]; x[4] = ROTL32(x[4], 7); x[21] = x[21] + x[5]; x[5] = ROTL32(x[5], 7);
		x[22] = x[22] + x[6]; x[6] = ROTL32(x[6], 7); x[23] = x[23] + x[7]; x[7] = ROTL32(x[7], 7);
		x[24] = x[24] + x[8]; x[8] = ROTL32(x[8], 7); x[25] = x[25] + x[9]; x[9] = ROTL32(x[9], 7);
		x[26] = x[26] + x[10]; x[10] = ROTL32(x[10], 7); x[27] = x[27] + x[11]; x[11] = ROTL32(x[11], 7);
		x[28] = x[28] + x[12]; x[12] = ROTL32(x[12], 7); x[29] = x[29] + x[13]; x[13] = ROTL32(x[13], 7);
		x[30] = x[30] + x[14]; x[14] = ROTL32(x[14], 7); x[31] = x[31] + x[15]; x[15] = ROTL32(x[15], 7);
		/* "swap x_00klm with x_01klm" */
		SWAP(x[0], x[8]); x[0] ^= x[16]; x[8] ^= x[24]; SWAP(x[1], x[9]); x[1] ^= x[17]; x[9] ^= x[25];
		SWAP(x[2], x[10]); x[2] ^= x[18]; x[10] ^= x[26]; SWAP(x[3], x[11]); x[3] ^= x[19]; x[11] ^= x[27];
		SWAP(x[4], x[12]); x[4] ^= x[20]; x[12] ^= x[28]; SWAP(x[5], x[13]); x[5] ^= x[21]; x[13] ^= x[29];
		SWAP(x[6], x[14]); x[6] ^= x[22]; x[14] ^= x[30]; SWAP(x[7], x[15]); x[7] ^= x[23]; x[15] ^= x[31];
		/* "swap x_1jk0m with x_1jk1m" */
		SWAP(x[16], x[18]); SWAP(x[17], x[19]); SWAP(x[20], x[22]); SWAP(x[21], x[23]); SWAP(x[24], x[26]); SWAP(x[25], x[27]); SWAP(x[28], x[30]); SWAP(x[29], x[31]);
		/* "add x_0jklm into x_1jklm modulo 2^32 rotate x_0jklm upwards by 11 bits" */
		x[16] = x[16] + x[0]; x[0] = ROTL32(x[0], 11); x[17] = x[17] + x[1]; x[1] = ROTL32(x[1], 11);
		x[18] = x[18] + x[2]; x[2] = ROTL32(x[2], 11); x[19] = x[19] + x[3]; x[3] = ROTL32(x[3], 11);
		x[20] = x[20] + x[4]; x[4] = ROTL32(x[4], 11); x[21] = x[21] + x[5]; x[5] = ROTL32(x[5], 11);
		x[22] = x[22] + x[6]; x[6] = ROTL32(x[6], 11); x[23] = x[23] + x[7]; x[7] = ROTL32(x[7], 11);
		x[24] = x[24] + x[8]; x[8] = ROTL32(x[8], 11); x[25] = x[25] + x[9]; x[9] = ROTL32(x[9], 11);
		x[26] = x[26] + x[10]; x[10] = ROTL32(x[10], 11); x[27] = x[27] + x[11]; x[11] = ROTL32(x[11], 11);
		x[28] = x[28] + x[12]; x[12] = ROTL32(x[12], 11); x[29] = x[29] + x[13]; x[13] = ROTL32(x[13], 11);
		x[30] = x[30] + x[14]; x[14] = ROTL32(x[14], 11); x[31] = x[31] + x[15]; x[15] = ROTL32(x[15], 11);
		/* "swap x_0j0lm with x_0j1lm" */
		SWAP(x[0], x[4]); x[0] ^= x[16]; x[4] ^= x[20]; SWAP(x[1], x[5]); x[1] ^= x[17]; x[5] ^= x[21];
		SWAP(x[2], x[6]); x[2] ^= x[18]; x[6] ^= x[22]; SWAP(x[3], x[7]); x[3] ^= x[19]; x[7] ^= x[23];
		SWAP(x[8], x[12]); x[8] ^= x[24]; x[12] ^= x[28]; SWAP(x[9], x[13]); x[9] ^= x[25]; x[13] ^= x[29];
		SWAP(x[10], x[14]); x[10] ^= x[26]; x[14] ^= x[30]; SWAP(x[11], x[15]); x[11] ^= x[27]; x[15] ^= x[31];
		/* "swap x_1jkl0 with x_1jkl1" */
		SWAP(x[16], x[17]); SWAP(x[18], x[19]); SWAP(x[20], x[21]); SWAP(x[22], x[23]); SWAP(x[24], x[25]); SWAP(x[26], x[27]); SWAP(x[28], x[29]); SWAP(x[30], x[31]);
	}
}

__host__
void skunk_cpu_setBlock_80(void *pdata)
{
	uint64_t message[20];
	memcpy(&message[0], pdata, 80);

	uint64_t p[8];
	uint64_t h[9];
	uint64_t t0, t1, t2;

	h[0] = 0x4903ADFF749C51CEull;
	h[1] = 0x0D95DE399746DF03ull;
	h[2] = 0x8FD1934127C79BCEull;
	h[3] = 0x9A255629FF352CB1ull;
	h[4] = 0x5DB62599DF6CA7B0ull;
	h[5] = 0xEABE394CA9D5C3F4ull;
	h[6] = 0x991112C71A75B523ull;
	h[7] = 0xAE18A40B660FCC33ull;
	// h[8] = h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[4] ^ h[5] ^ h[6] ^ h[7] ^ SPH_C64(0x1BD11BDAA9FC1A22);
	h[8] = 0xcab2076d98173ec4ULL;

	t0 = 64; // ptr
	t1 = 0x7000000000000000ull;
	t2 = 0x7000000000000040ull;

	memcpy(&p[0], &message[0], 64);

	TFBIG_4e_PRE(0);
	TFBIG_4o_PRE(1);
	TFBIG_4e_PRE(2);
	TFBIG_4o_PRE(3);
	TFBIG_4e_PRE(4);
	TFBIG_4o_PRE(5);
	TFBIG_4e_PRE(6);
	TFBIG_4o_PRE(7);
	TFBIG_4e_PRE(8);
	TFBIG_4o_PRE(9);
	TFBIG_4e_PRE(10);
	TFBIG_4o_PRE(11);
	TFBIG_4e_PRE(12);
	TFBIG_4o_PRE(13);
	TFBIG_4e_PRE(14);
	TFBIG_4o_PRE(15);
	TFBIG_4e_PRE(16);
	TFBIG_4o_PRE(17);
	TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

	message[10] = message[0] ^ p[0];
	message[11] = message[1] ^ p[1];
	message[12] = message[2] ^ p[2];
	message[13] = message[3] ^ p[3];
	message[14] = message[4] ^ p[4];
	message[15] = message[5] ^ p[5];
	message[16] = message[6] ^ p[6];
	message[17] = message[7] ^ p[7];

	message[18] = t2;

	uint64_t buffer[128];

	//	buffer[ 0] = message[ 8];
	buffer[0] = message[9];
	h[0] = buffer[1] = message[10];
	h[1] = buffer[2] = message[11];
	h[2] = buffer[3] = message[12];
	h[3] = buffer[4] = message[13];
	h[4] = buffer[5] = message[14];
	h[5] = buffer[6] = message[15];
	h[6] = buffer[7] = message[16];
	h[7] = buffer[8] = message[17];
	h[8] = buffer[9] = h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[4] ^ h[5] ^ h[6] ^ h[7] ^ 0x1BD11BDAA9FC1A22ULL;

	t0 = 0x50ull;
	t1 = 0xB000000000000000ull;
	t2 = t0^t1;

	p[0] = message[8] + h[0];	p[2] = h[2];		p[3] = h[3];	p[4] = h[4];
	p[5] = h[5] + t0;		p[6] = h[6] + t1;	p[7] = h[7];

	p[2] += p[3];	p[4] += p[5];	p[6] += p[7];

	p[3] = ROTL64(p[3], 36) ^ p[2];	p[5] = ROTL64(p[5], 19) ^ p[4];	p[7] = ROTL64(p[7], 37) ^ p[6];

	p[4] += p[7];	p[6] += p[5];

	p[7] = ROTL64(p[7], 27) ^ p[4];
	p[5] = ROTL64(p[5], 14) ^ p[6];

	buffer[10] = p[0];
	buffer[11] = p[2];
	buffer[12] = p[3];
	buffer[13] = p[4];
	buffer[14] = p[5];
	buffer[15] = p[6];
	buffer[16] = p[7];
	buffer[17] = ROTL64(p[3], 42);
	buffer[18] = ROTL64(p[5], 36);
	buffer[19] = ROTL64(p[7], 39);

	buffer[20] = h[6] + t1;
	buffer[21] = h[8] + 1;
	buffer[22] = h[7] + t2;
	buffer[23] = h[0] + 2;
	buffer[24] = h[8] + t0;
	buffer[25] = h[1] + 3;
	buffer[26] = h[0] + t1;
	buffer[27] = h[2] + 4;
	buffer[28] = h[1] + t2;
	buffer[29] = h[3] + 5;
	buffer[30] = h[2] + t0;
	buffer[31] = h[4] + 6;
	buffer[32] = h[3] + t1;
	buffer[33] = h[5] + 7;
	buffer[34] = h[4] + t2;
	buffer[35] = h[6] + 8;
	buffer[36] = h[5] + t0;
	buffer[37] = h[7] + 9;
	buffer[38] = h[6] + t1;
	buffer[39] = h[8] + 10;
	buffer[40] = h[7] + t2;
	buffer[41] = h[0] + 11;
	buffer[42] = h[8] + t0;
	buffer[43] = h[1] + 12;
	buffer[44] = h[0] + t1;
	buffer[45] = h[2] + 13;
	buffer[46] = h[1] + t2;
	buffer[47] = h[3] + 14;
	buffer[48] = h[2] + t0;
	buffer[49] = h[4] + 15;
	buffer[50] = h[3] + t1;
	buffer[51] = h[5] + 16;
	buffer[52] = h[4] + t2;
	buffer[53] = h[6] + 17;
	buffer[54] = h[5] + t0;
	buffer[55] = h[7] + 18;
	buffer[56] = h[6] + t1;

	buffer[57] = message[8];

	cudaMemcpyToSymbol(c_buffer, buffer, sizeof(buffer), 0, cudaMemcpyHostToDevice);

	CUDA_SAFE_CALL(cudaGetLastError());
}

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(TPB52, 3)
#else
__launch_bounds__(TPB50, 5)
#endif
void skunk_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint64_t *output64)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread >= threads)
		return;

	// quark_skein512
	uint2 h[9];
	uint2 t0, t1, t2;

	uint32_t nonce = cuda_swab32(startNounce + thread);
	uint2 nonce2 = make_uint2(c_buffer[0].x, nonce);

	uint2 p[8];
	p[1] = nonce2;

	h[0] = c_buffer[1];
	h[1] = c_buffer[2];
	h[2] = c_buffer[3];
	h[3] = c_buffer[4];
	h[4] = c_buffer[5];
	h[5] = c_buffer[6];
	h[6] = c_buffer[7];
	h[7] = c_buffer[8];
	h[8] = c_buffer[9];

	t0 = vectorize(0x50ull);
	t1 = vectorize(0xB000000000000000ull);
	t2 = t0^t1;

	p[1] = nonce2 + h[1];	p[0] = c_buffer[10] + p[1];
	p[2] = c_buffer[11];
	p[3] = c_buffer[12];
	p[4] = c_buffer[13];
	p[5] = c_buffer[14];
	p[6] = c_buffer[15];
	p[7] = c_buffer[16];

	//		TFBIGMIX8e();
	p[1] = ROL2(p[1], 46) ^ p[0];
	p[2] += p[1];
	p[0] += p[3];
	p[1] = ROL2(p[1], 33) ^ p[2];
	p[3] = c_buffer[17] ^ p[0];
	p[4] += p[1];
	p[6] += p[3];
	p[0] += p[5];
	p[2] += p[7];
	p[1] = ROL2(p[1], 17) ^ p[4];
	p[3] = ROL2(p[3], 49) ^ p[6];
	p[5] = c_buffer[18] ^ p[0];
	p[7] = c_buffer[19] ^ p[2];
	p[6] += p[1];
	p[0] += p[7];
	p[2] += p[5];
	p[4] += p[3];
	p[1] = ROL2(p[1], 44) ^ p[6];
	p[7] = ROL2(p[7], 9) ^ p[0];
	p[5] = ROL2(p[5], 54) ^ p[2];
	p[3] = ROR8(p[3]) ^ p[4];

	p[0] += h[1];	p[1] += h[2];	p[2] += h[3];	p[3] += h[4];	p[4] += h[5];	p[5] += c_buffer[20];	p[7] += c_buffer[21];	p[6] += c_buffer[22];
	TFBIGMIX8o();
	p[0] += h[2];	p[1] += h[3];	p[2] += h[4];	p[3] += h[5];	p[4] += h[6];	p[5] += c_buffer[22];	p[7] += c_buffer[23];	p[6] += c_buffer[24];
	TFBIGMIX8e();
	p[0] += h[3];	p[1] += h[4];	p[2] += h[5];	p[3] += h[6];	p[4] += h[7];	p[5] += c_buffer[24];	p[7] += c_buffer[25];	p[6] += c_buffer[26];
	TFBIGMIX8o();
	p[0] += h[4];	p[1] += h[5];	p[2] += h[6];	p[3] += h[7];	p[4] += h[8];	p[5] += c_buffer[26];	p[7] += c_buffer[27];	p[6] += c_buffer[28];
	TFBIGMIX8e();
	p[0] += h[5];	p[1] += h[6];	p[2] += h[7];	p[3] += h[8];	p[4] += h[0];	p[5] += c_buffer[28];	p[7] += c_buffer[29];	p[6] += c_buffer[30];
	TFBIGMIX8o();
	p[0] += h[6];	p[1] += h[7];	p[2] += h[8];	p[3] += h[0];	p[4] += h[1];	p[5] += c_buffer[30];	p[7] += c_buffer[31];	p[6] += c_buffer[32];
	TFBIGMIX8e();
	p[0] += h[7];	p[1] += h[8];	p[2] += h[0];	p[3] += h[1];	p[4] += h[2];	p[5] += c_buffer[32];	p[7] += c_buffer[33];	p[6] += c_buffer[34];
	TFBIGMIX8o();
	p[0] += h[8];	p[1] += h[0];	p[2] += h[1];	p[3] += h[2];	p[4] += h[3];	p[5] += c_buffer[34];	p[7] += c_buffer[35];	p[6] += c_buffer[36];
	TFBIGMIX8e();
	p[0] += h[0];	p[1] += h[1];	p[2] += h[2];	p[3] += h[3];	p[4] += h[4];	p[5] += c_buffer[36];	p[7] += c_buffer[37];	p[6] += c_buffer[38];
	TFBIGMIX8o();
	p[0] += h[1];	p[1] += h[2];	p[2] += h[3];	p[3] += h[4];	p[4] += h[5];	p[5] += c_buffer[38];	p[7] += c_buffer[39];	p[6] += c_buffer[40];
	TFBIGMIX8e();
	p[0] += h[2];	p[1] += h[3];	p[2] += h[4];	p[3] += h[5];	p[4] += h[6];	p[5] += c_buffer[40];	p[7] += c_buffer[41];	p[6] += c_buffer[42];
	TFBIGMIX8o();
	p[0] += h[3];	p[1] += h[4];	p[2] += h[5];	p[3] += h[6];	p[4] += h[7];	p[5] += c_buffer[42];	p[7] += c_buffer[43];	p[6] += c_buffer[44];
	TFBIGMIX8e();
	p[0] += h[4];	p[1] += h[5];	p[2] += h[6];	p[3] += h[7];	p[4] += h[8];	p[5] += c_buffer[44];	p[7] += c_buffer[45];	p[6] += c_buffer[46];
	TFBIGMIX8o();
	p[0] += h[5];	p[1] += h[6];	p[2] += h[7];	p[3] += h[8];	p[4] += h[0];	p[5] += c_buffer[46];	p[7] += c_buffer[47];	p[6] += c_buffer[48];
	TFBIGMIX8e();
	p[0] += h[6];	p[1] += h[7];	p[2] += h[8];	p[3] += h[0];	p[4] += h[1];	p[5] += c_buffer[48];	p[7] += c_buffer[49];	p[6] += c_buffer[50];
	TFBIGMIX8o();
	p[0] += h[7];	p[1] += h[8];	p[2] += h[0];	p[3] += h[1];	p[4] += h[2];	p[5] += c_buffer[50];	p[7] += c_buffer[51];	p[6] += c_buffer[52];
	TFBIGMIX8e();
	p[0] += h[8];	p[1] += h[0];	p[2] += h[1];	p[3] += h[2];	p[4] += h[3];	p[5] += c_buffer[52];	p[7] += c_buffer[53];	p[6] += c_buffer[54];
	TFBIGMIX8o();
	p[0] += h[0];	p[1] += h[1];	p[2] += h[2];	p[3] += h[3];	p[4] += h[4];	p[5] += c_buffer[54];	p[7] += c_buffer[55];	p[6] += c_buffer[56];

	p[0] ^= c_buffer[57];
	p[1] ^= nonce2;

	t0 = vectorize(8); // extra
	t1 = vectorize(0xFF00000000000000ull); // etype
	t2 = t0^t1;

	h[0] = p[0];
	h[1] = p[1];
	h[2] = p[2];
	h[3] = p[3];
	h[4] = p[4];
	h[5] = p[5];
	h[6] = p[6];
	h[7] = p[7];

	h[8] = h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[4] ^ h[5] ^ h[6] ^ h[7] ^ vectorize(0x1BD11BDAA9FC1A22);
	p[0] = p[1] = p[2] = p[3] = p[4] = p[5] = p[6] = p[7] = vectorize(0);

	TFBIG_4e_UI2(0);
	TFBIG_4o_UI2(1);
	TFBIG_4e_UI2(2);
	TFBIG_4o_UI2(3);
	TFBIG_4e_UI2(4);
	TFBIG_4o_UI2(5);
	TFBIG_4e_UI2(6);
	TFBIG_4o_UI2(7);
	TFBIG_4e_UI2(8);
	TFBIG_4o_UI2(9);
	TFBIG_4e_UI2(10);
	TFBIG_4o_UI2(11);
	TFBIG_4e_UI2(12);
	TFBIG_4o_UI2(13);
	TFBIG_4e_UI2(14);
	TFBIG_4o_UI2(15);
	TFBIG_4e_UI2(16);
	TFBIG_4o_UI2(17);
	TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

	// resulting hash in p[0]..p[7]

	// x11_cubehash512
	uint2x4 *pvec = (uint2x4*)p;

	uint32_t x[32] = {
		0x2AEA2A61, 0x50F494D4, 0x2D538B8B, 0x4167D83E,
		0x3FEE2313, 0xC701CF8C, 0xCC39968E, 0x50AC5695,
		0x4D42C787, 0xA647A8B3, 0x97CF0BEF, 0x825B4537,
		0xEEF864D2, 0xF22090C4, 0xD0E5CD33, 0xA23911AE,
		0xFCD398D9, 0x148FE485, 0x1B017BEF, 0xB6444532,
		0x6A536159, 0x2FF5781C, 0x91FA7934, 0x0DBADEA9,
		0xD65C8A2B, 0xA5A70E75, 0xB1C62456, 0xBC796576,
		0x1921C8F7, 0xE7989AF1, 0x7795D246, 0xD43E3B44
	};

	// erste Hälfte des Hashes (32 bytes)
	//Update32(x, (const BitSequence*)Hash);
	*(uint2x4*)&x[0] ^= pvec[0];

	rrounds(x);

	// zweite Hälfte des Hashes (32 bytes)
	//        Update32(x, (const BitSequence*)(Hash+8));
	*(uint2x4*)&x[0] ^= pvec[1];

	rrounds(x);

	// Padding Block
	x[0] ^= 0x80;
	rrounds(x);

	//	Final(x, (BitSequence*)Hash);
	x[31] ^= 1;

	/* "the state is then transformed invertibly through 10r identical rounds" */
	#pragma unroll 10
	for (int i = 0; i < 10; ++i)
		rrounds(x);

	/* "output the first h/8 bytes of the state" */
	uint32_t *Hash = (uint32_t*)&output64[thread << 3];
	*(uint2x4*)&Hash[0] = *(uint2x4*)&x[0];
	*(uint2x4*)&Hash[8] = *(uint2x4*)&x[8];
}

void skunk_cpu_hash(int thr_id, uint32_t throughput, uint32_t startNounce, uint32_t *d_hash, uint32_t* d_resNonce)
{
	uint32_t tpb = TPB52;
	int dev_id = device_map[thr_id];

	if (device_sm[dev_id] <= 500)
		tpb = TPB50;

	const dim3 grid((throughput + tpb - 1) / tpb);
	const dim3 block(tpb);

	skunk_gpu_hash_80 <<<grid, block>>> (throughput, startNounce, (uint64_t*)d_hash);
	x13_fugue512_cpu_hash_64(thr_id, throughput, d_hash);
	streebog_cpu_hash_64_final(thr_id, throughput, d_hash, d_resNonce);
}

extern "C" int scanhash_skunk(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	int dev_id = device_map[thr_id];

	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	uint32_t throughput = cuda_default_throughput(thr_id, 1 << 21); // 256*256*16
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x00FF;

	const uint64_t highTarget = *(uint64_t*)&ptarget[6];

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}

		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		x13_fugue512_cpu_init(thr_id, throughput);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 8 * sizeof(uint64_t) * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
		h_resNonce[thr_id] = (uint32_t*)malloc(NBN * sizeof(uint32_t));
		if (h_resNonce[thr_id] == NULL) {
			gpulog(LOG_ERR, thr_id, "Host memory allocation failed");
			exit(EXIT_FAILURE);
		}
		init[thr_id] = true;
	}

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	streebog_set_target(ptarget);

	skunk_cpu_setBlock_80(endiandata);

	cudaMemset(d_resNonce[thr_id], 0xff, NBN * sizeof(uint32_t));
	int rc = 0;
	do {
		// Hash with CUDA
		skunk_cpu_hash(thr_id, throughput, pdata[19], d_hash[thr_id], d_resNonce[thr_id]);
		cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		if (h_resNonce[thr_id][0] != UINT32_MAX) {
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNounce = pdata[19];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], startNounce + h_resNonce[thr_id][0]);
			skunk_hash(vhash64, endiandata);
			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget)) {
				rc = 1;
				*hashes_done = pdata[19] - first_nonce + throughput;
				work_set_target_ratio(work, vhash64);
				pdata[19] = startNounce + h_resNonce[thr_id][0];
				if (h_resNonce[thr_id][1] != UINT32_MAX) {
					be32enc(&endiandata[19], startNounce + h_resNonce[thr_id][1]);
					skunk_hash(vhash64, endiandata);
					if (bn_hash_target_ratio(vhash64, ptarget) > work->shareratio[0])
						work_set_target_ratio(work, vhash64);
					pdata[21] = startNounce + h_resNonce[thr_id][1];
					rc = 2;
				}
				return rc;
			}
			else {
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", h_resNonce[thr_id][0]);
				cudaMemset(d_resNonce[thr_id], 0xff, NBN * sizeof(uint32_t));
			}
		}
		pdata[19] += throughput;
	} while (!work_restart[thr_id].restart && (uint64_t)max_nonce > (uint64_t)throughput + (uint64_t)pdata[19]);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_skunk(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	x13_fugue512_cpu_free(thr_id);
	cudaFree(d_hash[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
