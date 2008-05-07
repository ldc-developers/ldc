#include <list>
#include <iostream>
#include <limits>
#include <cmath>
#include <cstdlib>

using namespace std;

numeric_limits<double> real;
double delta = sqrt(real.epsilon()), infinity = real.infinity();

struct Vec {
  double x, y, z;
  Vec(double x2, double y2, double z2) : x(x2), y(y2), z(z2) {}
};
Vec operator+(const Vec &a, const Vec &b)
{ return Vec(a.x+b.x, a.y+b.y, a.z+b.z); }
Vec operator-(const Vec &a, const Vec &b)
{ return Vec(a.x-b.x, a.y-b.y, a.z-b.z); }
Vec operator*(double a, const Vec &b) { return Vec(a*b.x, a*b.y, a*b.z); }
double dot(const Vec &a, const Vec &b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
Vec unitise(const Vec &a) { return (1 / sqrt(dot(a, a))) * a; }

typedef pair<double, Vec> Hit;

struct Ray {
  Vec orig, dir;
  Ray(const Vec &o, const Vec &d) : orig(o), dir(d) {}
};

struct Scene {
  virtual ~Scene() {};
  virtual void intersect(Hit &, const Ray &) const = 0;
};

struct Sphere : public Scene {
  Vec center;
  double radius;

  Sphere(Vec c, double r) : center(c), radius(r) {}
  ~Sphere() {}

  double ray_sphere(const Ray &ray) const {
    Vec v = center - ray.orig;
    double b = dot(v, ray.dir), disc = b*b - dot(v, v) + radius * radius;
    if (disc < 0) return infinity;
    double d = sqrt(disc), t2 = b + d;
    if (t2 < 0) return infinity;
    double t1 = b - d;
    return (t1 > 0 ? t1 : t2);
  }

  void intersect(Hit &hit, const Ray &ray) const {
    double lambda = ray_sphere(ray);
    if (lambda >= hit.first) return;
    hit = Hit(lambda, unitise(ray.orig + lambda*ray.dir - center));
  }
};

typedef list<Scene *> Scenes;
struct Group : public Scene {
  Sphere bound;
  Scenes child;

  Group(Sphere b, Scenes c) : bound(b), child(c) {}
  ~Group() {
    for (Scenes::const_iterator it=child.begin(); it!=child.end(); ++it)
      delete *it;
  }

  void intersect(Hit &hit, const Ray &ray) const {
    double l = bound.ray_sphere(ray);
    if (l < hit.first)
      for (Scenes::const_iterator it=child.begin(); it!=child.end(); ++it)
        (*it)->intersect(hit, ray);
  }
};

Hit intersect(const Ray &ray, const Scene &s) {
  Hit res = Hit(infinity, Vec(0, 0, 0));
  s.intersect(res, ray);
  return res;
}

double ray_trace(const Vec &light, const Ray &ray, const Scene &s) {
  Hit hit = intersect(ray, s);
  if (hit.first == infinity) return 0;
  double g = dot(hit.second, light);
  if (g >= 0) return 0.;
  Vec p = ray.orig + hit.first*ray.dir + delta*hit.second;
  return (intersect(Ray(p, -1. * light), s).first < infinity ? 0 : -g);
}

Scene *create(int level, const Vec &c, double r) {
  Scene *s = new Sphere(c, r);
  if (level == 1) return s;
  Scenes child;
  child.push_back(s);
  double rn = 3*r/sqrt(12.);
  for (int dz=-1; dz<=1; dz+=2)
    for (int dx=-1; dx<=1; dx+=2)
      child.push_back(create(level-1, c + rn*Vec(dx, 1, dz), r/2));
  return new Group(Sphere(c, 3*r), child);
}

int main(int argc, char *argv[]) {
  int level = 6, n = 512, ss = 4;
  if (argc == 2) level = atoi(argv[1]);
  Vec light = unitise(Vec(-1, -3, 2));
  Scene *s(create(level, Vec(0, -1, 0), 1));
  cout << "P5\n" << n << " " << n << "\n255\n";
  for (int y=n-1; y>=0; --y)
    for (int x=0; x<n; ++x) {
      double g=0;
      for (int dx=0; dx<ss; ++dx)
        for (int dy=0; dy<ss; ++dy) {
          Vec dir(unitise(Vec(x+dx*1./ss-n/2., y+dy*1./ss-n/2., n)));
          g += ray_trace(light, Ray(Vec(0, 0, -4), dir), *s);
        }
      cout << char(int(.5 + 255. * g / (ss*ss)));
    }
  delete s;
  return 0;
}
