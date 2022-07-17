const float M_PI = 3.14;
const float M_2PI	=	6.283185307179586476925286766559005768;


float tr(mat2 S) {
  return S[0][0] + S[1][1];
}

vec2 eig(mat2 S) {
  float tr = tr(S);
  float rt = sqrt(tr*tr - 4 * determinant(S));
  return vec2(tr + rt, tr - rt) / 2;
}


float madd(const float a, const float b, const float c) {
	return a * b + c;
}


float safe_sqrt(float f) {
  return sqrt(max(0.0, f));
}


float fast_erff(float x) {
	/* Examined 1082130433 values of erff on [0,4]: 1.93715e-06 max error. */
	/* Abramowitz and Stegun, 7.1.28. */
	const float a1 = 0.0705230784f;
	const float a2 = 0.0422820123f;
	const float a3 = 0.0092705272f;
	const float a4 = 0.0001520143f;
	const float a5 = 0.0002765672f;
	const float a6 = 0.0000430638f;
	const float a = abs(x);
	if(a >= 12.3f) {
		return sign(x);
	}
	const float b = 1.0f - (1.0f - a);  /* Crush denormals. */
	const float r = madd(madd(madd(madd(madd(madd(a6, b, a5), b, a4), b, a3), b, a2), b, a1), b, 1.0f);
	const float s = r * r;  /* ^2 */
	const float t = s * s;  /* ^4 */
	const float u = t * t;  /* ^8 */
	const float v = u * u;  /* ^16 */
	return sign(x) * (1.0f - 1.0f / v);
}


float fast_ierff(float x) {
	/* From: Approximating the erfinv function by Mike Giles. */
	/* To avoid trouble at the limit, clamp input to 1-eps. */
	x = clamp(x, 0.0, 1.0-0.00000000001);
	float a = abs(x);
	if(a > 0.99999994f) {
		a = 0.99999994f;
	}
	float w = -log((1.0f - a) * (1.0f + a)), p;
	if(w < 5.0f) {
		w = w - 2.5f;
		p =  2.81022636e-08f;
		p = madd(p, w,  3.43273939e-07f);
		p = madd(p, w, -3.5233877e-06f);
		p = madd(p, w, -4.39150654e-06f);
		p = madd(p, w,  0.00021858087f);
		p = madd(p, w, -0.00125372503f);
		p = madd(p, w, -0.00417768164f);
		p = madd(p, w,  0.246640727f);
		p = madd(p, w,  1.50140941f);
	}
	else {
		w = sqrt(w) - 3.0f;
		p = -0.000200214257f;
		p = madd(p, w,  0.000100950558f);
		p = madd(p, w,  0.00134934322f);
		p = madd(p, w, -0.00367342844f);
		p = madd(p, w,  0.00573950773f);
		p = madd(p, w, -0.0076224613f);
		p = madd(p, w,  0.00943887047f);
		p = madd(p, w,  1.00167406f);
		p = madd(p, w,  2.83297682f);
	}
	return p * x;
}
