#pragma once

namespace DirectX
{
struct XMFLOAT2
{
	float x, y;
	XMFLOAT2() : x(0.0f), y(0.0f) {}
	XMFLOAT2(float x_, float y_) : x(x_), y(y_) {}
};

struct XMFLOAT3
{
	float x, y, z;
	XMFLOAT3() : x(0.0f), y(0.0f), z(0.0f) {}
	XMFLOAT3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct XMFLOAT4
{
	float x, y, z, w;
	XMFLOAT4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
	XMFLOAT4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
};
}
