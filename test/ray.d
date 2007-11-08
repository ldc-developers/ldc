//import std.stdio, std.math, std.string;
//import tools.base;

int atoi(char[] s) {
    int i, fac=1;
    bool neg = (s.length) && (s[0] == '-');
    char[] a = neg ? s[1..$] : s;
    foreach_reverse(c; a) {
        i += (c-'0') * fac;
        fac *= 10;
    }
    return !neg ? i : -i;
}

pragma(LLVM_internal, "intrinsic", "llvm.sqrt.f64")
double sqrt(double val);

double delta;
static this() { delta=sqrt(real.epsilon); }

struct Vec {
  double x, y, z;
  Vec opAdd(ref Vec other) { return Vec(x+other.x, y+other.y, z+other.z); }
  Vec opSub(ref Vec other) { return Vec(x-other.x, y-other.y, z-other.z); }
  Vec opMul(double a) { return Vec(x*a, y*a, z*a); }
  double dot(ref Vec other) { return x*other.x+y*other.y+z*other.z; }
  Vec unitise() { return opMul(1.0/sqrt(dot(*this))); }
}

struct Pair(T, U) { T first; U second; }
typedef Pair!(double, Vec) Hit;

struct Ray { Vec orig, dir; }

class Scene {
  //abstract void intersect(ref Hit, ref Ray);
  void intersect(ref Hit, ref Ray) {}
}

class Sphere : Scene {
  Vec center;
  double radius;
  //mixin This!("center, radius");
  this(ref Vec c, double r)
  {
    center = c;
    radius = r;
  }
  double ray_sphere(ref Ray ray) {
    auto v = center - ray.orig, b = v.dot(ray.dir), disc=b*b - v.dot(v) + radius*radius;
    if (disc < 0) return double.infinity;
    auto d = sqrt(disc), t2 = b + d;
    if (t2 < 0) return double.infinity;
    auto t1 = b - d;
    return (t1 > 0 ? t1 : t2);
  }
  void intersect(ref Hit hit, ref Ray ray) {
    auto lambda = ray_sphere(ray);
    if (lambda < hit.first)
      hit = Hit(lambda, (ray.orig + lambda*ray.dir - center).unitise);
  }
}

class Group : Scene {
  Sphere bound;
  Scene[] children;
  //mixin This!("bound, children");
  this (Sphere s, Scene[] c)
  {
    bound = s;
    children = c;
  }
  void intersect(ref Hit hit, ref Ray ray) {
    auto l = bound.ray_sphere(ray);
    if (l < hit.first) foreach (child; children) child.intersect(hit, ray);
  }
}

double ray_trace(ref Vec light, ref Ray ray, Scene s) {
  auto hit=Hit(double.infinity, Vec(0, 0, 0));
  s.intersect(hit, ray);
  if (hit.first == double.infinity) return 0.0;
  auto g = hit.second.dot(light);
  if (g >= 0) return 0.0;
  auto p = ray.orig + ray.dir*hit.first + hit.second*delta;
  auto hit2=Hit(double.infinity, Vec(0, 0, 0));
  s.intersect(hit2, Ray(p, light*-1.0));
  return (hit2.first < double.infinity ? 0 : -g);
}

Scene create(int level, ref Vec c, double r) {
  auto s = new Sphere(c, r);
  if (level == 1) return s;
  Scene[] children=[s];
  double rn = 3*r/sqrt(12.0);
  for (int dz=-1; dz<=1; dz+=2)
    for (int dx=-1; dx<=1; dx+=2)
      children~=create(level-1, c + Vec(dx, 1, dz)*rn, r/2);
  return new Group(new Sphere(c, 3*r), children);
}

void main(string[] args) {
  int level = (args.length==3 ? args[1].atoi() : 9),
    n = (args.length==3 ? args[2].atoi() : 512), ss = 4;
  auto light = Vec(-1, -3, 2).unitise();
  auto s=create(level, Vec(0, -1, 0), 1);
  printf("P5\n%d %d\n255", n, n);
  for (int y=n-1; y>=0; --y)
    for (int x=0; x<n; ++x) {
      double g=0;
      for (int d=0; d<ss*ss; ++d) {
        auto dir=Vec(x+(d%ss)*1.0/ss-n/2.0, y+(d/ss)*1.0/ss-n/2.0, n).unitise();
    g += ray_trace(light, Ray(Vec(0, 0, -4), dir), s);
      }
      printf("%c", cast(ubyte)(0.5 + 255.0 * g / (ss*ss)));
    }
}
