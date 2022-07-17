/**
 * From https://gist.github.com/msbarry/cd98f928542f5152111a
 */


vec3 rgb2hsv(vec3 c) {
  vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4 p = c.g < c.b ? vec4(c.bg, K.wz) : vec4(c.gb, K.xy);
  vec4 q = c.r < p.x ? vec4(p.xyw, c.r) : vec4(c.r, p.yzx);

  float d = q.x - min(q.w, q.y);
  float e = 1.0e-10;
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 rgb2xyz(vec3 c) {
  float R = ((c.r > 0.04045) ? pow((( c.r + 0.055 ) / 1.055), 2.4) : (c.r / 12.92)) * 100.0;
  float G = ((c.g > 0.04045) ? pow((( c.g + 0.055 ) / 1.055), 2.4) : (c.g / 12.92)) * 100.0;
  float B = ((c.b > 0.04045) ? pow((( c.b + 0.055 ) / 1.055), 2.4) : (c.b / 12.92)) * 100.0;

  float X = R * 0.4124 + G * 0.3576 + B * 0.1805;
  float Y = R * 0.2126 + G * 0.7152 + B * 0.0722;
  float Z = R * 0.0193 + G * 0.1192 + B * 0.9505;

  return vec3(X, Y, Z);
}

vec3 xyz2rgb(vec3 c) {
  float X = c.x / 100.0;
  float Y = c.y / 100.0;
  float Z = c.z / 100.0;

  float R = X *  3.2406 + Y * -1.5372 + Z * -0.4986;
  float G = X * -0.9689 + Y *  1.8758 + Z *  0.0415;
  float B = X *  0.0557 + Y * -0.2040 + Z *  1.0570;

  R = ((R > 0.0031308) ? (1.055 * ( pow( R, 1./2.4 ) ) - 0.055) : (12.92 * R));
  G = ((G > 0.0031308) ? (1.055 * ( pow( G, 1./2.4 ) ) - 0.055) : (12.92 * G));
  B = ((B > 0.0031308) ? (1.055 * ( pow( B, 1./2.4 ) ) - 0.055) : (12.92 * B));

  return vec3(R, G, B);
}

vec3 xyz2lab(vec3 c) {
  float X = c.x / 95.047;
  float Y = c.y / 100.0;
  float Z = c.z / 108.883;

  X = ((X > 0.008856) ? (pow( X, 1./3.)) : (( 7.787 * X ) + ( 16./116.)));
  Y = ((Y > 0.008856) ? (pow( Y, 1./3.)) : (( 7.787 * Y ) + ( 16./116.)));
  Z = ((Z > 0.008856) ? (pow( Z, 1./3.)) : (( 7.787 * Z ) + ( 16./116.)));

  float L = ( 116. * Y ) - 16.;
  float a = 500. * ( X - Y );
  float b = 200. * ( Y - Z );

  return vec3(L, a, b);
}

vec3 lab2xyz(vec3 c) {
  float L = c.x;
  float a = c.y;
  float b = c.z;

  float Y = ( L + 16. ) / 116.;
  float X = a / 500. + Y;
  float Z = Y - b / 200.;

  Y = ((pow(Y,3.) > 0.008856) ? (pow(Y,3.)) : ((Y - 16. / 116.) / 7.787));
  X = ((pow(X,3.) > 0.008856) ? (pow(X,3.)) : ((X - 16. / 116.) / 7.787));
  Z = ((pow(Z,3.) > 0.008856) ? (pow(Z,3.)) : ((Z - 16. / 116.) / 7.787));

  float ref_X = 95.047;
  float ref_Y = 100.0;
  float ref_Z = 108.883;

  return vec3(ref_X * X, ref_Y * Y, ref_Z * Z);
}


vec3 lab2rgb(vec3 c) {
  return xyz2rgb(lab2xyz(c));
}


vec3 rgb2lab(vec3 c) {
  return xyz2lab(rgb2xyz(c));
}
