#pragma once

#define F32_EXP_MAX_X  88.722839111671f
#define F32_EXP_MIN_X -103.972084044388f

#define F16_EXP_MAX_X __float2half(11.089866f)
#define F16_EXP_MIN_X __float2half(-16.635532f)

#define F16_ONE __float2half(1.0f)

#define FLOAT4(x) (*reinterpret_cast<float4 *>(&(x)))

#define HALF2(x) (*reinterpret_cast<half2 *>(&(x)))

#define LD_ST_128BITS(x) (*reinterpret_cast<float4 *>(&(x)))

#define INT4(x) (*reinterpret_cast<int4 *>(&(x)))

/*
#define ROW_MAJOR(row, col, ld) ((row) * (ld) + (col))

#define COL_MAJOR(row, col, ld) ((col) * (ld) + (row))

#define LD_FLOAT4(x) (*reinterpret_cast<const float4 *>(&(x)))

#define ST_FLOAT4(x) (*reinterpret_cast<float4 *>(&(x)))

#define LD_HALF2(x) (*reinterpret_cast<const half2 *>(&(x)))

#define ST_HALF2(x) (*reinterpret_cast<half2 *>(&(x)))
*/




