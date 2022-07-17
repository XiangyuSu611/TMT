

/**
 * Cube faces are in the order +x, -x, +y, -x, +z, -z.
 * Input vec coordinates are between -1 and 1.
 */
vec3 cubemap_face_to_world(vec2 vec, int cube_face) {
  vec3 vec_world = normalize(vec3(vec.xy, 1) );
  switch(cube_face) {
    case 0: vec_world = normalize(vec3(1, vec.y, -vec.x)); break;
    case 1: vec_world = normalize(vec3(-1, vec.y, vec.x)); break;
    case 2: vec_world = normalize(vec3(vec.x, 1, -vec.y)); break;
    case 3: vec_world = normalize(vec3(vec.x, -1, vec.y)); break;
    case 4: vec_world = normalize(vec3(vec.xy, 1)); break;
    case 5: vec_world = normalize(vec3(-vec.x, vec.y, -1)); break;
  }
  return vec_world;
}
