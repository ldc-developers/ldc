// modified version of scrapple.qd to work with llvmdc
import std.stdio;

//version(none)
void main() {
  screen(640, 480);
  pset(10, 10);
  line(0, 0, 100, 100, Box, Back(Red~Black));
  for (int i=0; i<=100; i+=10) {
    line(i, 0, 100-i, 100);
    line(0, i, 100, 100-i);
  }
  circle(100, 100, 50, 15, White~Black, Fill=White~Black);
  paint(200, 200, Red, Back=White);
  circle(100, 100, 50, 15, White);
  paint(200, 200, Black);
  pset(10, 11); pset(10, 11, Black);
  pset(10, 10);
  SDL_Delay(5000);
}

extern(C) {
  struct SDL_Rect {
    short x, y;
    ushort w, h;
  }
  struct SDL_PixelFormat {
    //SDL_Palette *palette;
    void *palette;
    ubyte BitsPerPixel, BytesPerPixel, Rloss, Gloss, Bloss, Aloss, Rshift, Gshift, Bshift, Ashift;
    uint Rmask, Gmask, Bmask, Amask, colorkey; ubyte alpha;
  }
  struct SDL_Surface {
    uint flags;
    SDL_PixelFormat *format;
    int w, h;
    ushort pitch;
    void *pixels;
    int offset;
    void *hwdata;
    SDL_Rect clip_rect;
    uint unused;
    uint locked;
    void *map;
    uint format_version;
    int refcount;
  }
  uint SDL_MapRGBA(SDL_PixelFormat *format, ubyte r, ubyte g, ubyte b, ubyte a);
  void SDL_GetRGBA(uint pixel, SDL_PixelFormat *fmt, ubyte *r, ubyte *g, ubyte *b, ubyte *a);
  int SDL_LockSurface(SDL_Surface *);
  void SDL_UnlockSurface(SDL_Surface *);
  SDL_Surface * SDL_SetVideoMode(int width, int height, int bpp, uint flags);
  int SDL_Flip(SDL_Surface *);
  void SDL_Delay(uint);
  const uint SDL_SWSURFACE=0;
  const uint SDL_HWSURFACE=1;
  const uint SDL_DOUBLEBUF=0x40000000;
  const uint SDL_FULLSCREEN=0x80000000;
}

SDL_Surface *display;

void putpixel32(SDL_Surface *surf, int x, int y, ubyte[4] col) {
  uint *bufp = cast(uint *)surf.pixels + y*surf.pitch/4 + x;
  *bufp = SDL_MapRGBA(surf.format, col[0], col[1], col[2], col[3]);
}

void getpixel32(SDL_Surface *surf, int x, int y, ubyte[4] *col) {
  uint *bufp = cast(uint *)surf.pixels + y*surf.pitch/4 + x;
  SDL_GetRGBA(*bufp, surf.format, &(*col)[0], &(*col)[1], &(*col)[2], &(*col)[3]);
}

struct rgb {
  ubyte[3] values;
  rgb opCat(rgb other) {
    rgb res;
    foreach (id, ref v; res.values) v=(values[id]+other.values[id])/2;
    return res;
  }
  bool opEquals(rgb r) {
    return values == r.values;
  }
}

void putpixel(SDL_Surface *surf, int x, int y, rgb c) {
  if ( (x<0) || (y<0) || (x!<surf.w) || (y!<surf.h) ) return;
  putpixel32(surf, x, y, [c.values[0], c.values[1], c.values[2], 0]);
}

const rgb White={[255, 255, 255]};
const rgb Black={[0, 0, 0]};
const rgb Red={[255, 0, 0]};
const rgb Green={[0, 255, 0]};
const rgb Blue={[0, 0, 255]};
rgb color=White;
rgb back=Black;

template failfind(U, T...) {
  static if (T.length)
    static if (is(T[0] == U)) static assert(false, "Duplicate "~U.stringof~" found!");
    else const bool failfind=failfind!(U, T[1..$]);
  else
    const bool failfind=true;
}

template select(U, T...) {
  static if(T.length)
    static if (is(U == T[0])) { static if (failfind!(U, T[1..$])) { }; const int select = 0; }
    else
      static if (select!(U, T[1..$]) != -1)
        const int select = 1 + select!(U, T[1..$]);
      else
        const int select = -1;
  else
    const int select = -1;
}

typedef rgb back_rgb;
back_rgb Back(rgb r) { return cast(back_rgb) r; }
back_rgb Back() { return cast(back_rgb) back; }
typedef rgb box_rgb;
box_rgb Box(rgb r) { return cast(box_rgb) r; }
box_rgb Box() { return cast(box_rgb) color; }
alias Back Fill;

void execParams(T...)(T params) {
  const int col=select!(rgb, T);
  static if (col != -1) color=params[col];
  const int bcol=select!(back_rgb, T);
  static if (bcol != -1) back=cast(rgb) params[bcol];
  const int boxcol=select!(box_rgb, T);
  static if (boxcol != -1) color=cast(rgb) params[boxcol];
}

void pset(T...)(int x, int y, T params) {
  SDL_LockSurface(display);
  scope(exit) { SDL_UnlockSurface(display); SDL_Flip(display); }
  execParams(params);
  putpixel(display, x, y, color);
}

rgb pget(int x, int y) {
  SDL_LockSurface(display);
  scope(exit) SDL_UnlockSurface(display);
  ubyte[4] c;
  getpixel32(display, x, y, &c);
  rgb res; res.values[]=c[0..3]; return res;
}

void swap(T)(ref T a, ref T b) { T c=a; a=b; b=c; }

T abs(T)(T f) { return f < 0 ? -f : f; }

void bresenham(bool countUp=true, bool steep=false)(int x0, int y0, int x1, int y1) {
  auto deltax = x1 - x0, deltay = y1 - y0;
  static if (steep) {
    auto Δerror = cast(float)deltax / cast(float)deltay;
    auto var2 = x0;
    const string name="y";
  } else {
    auto Δerror = cast(float)deltay / cast(float)deltax;
    auto var2 = y0;
    const string name="x";
  }
  auto error = 0f;
  for (auto var1 = mixin(name~'0'); var1 <= mixin(name~'1'); ++var1) {
    static if (steep) putpixel(display, var2, var1, color);
    else putpixel(display, var1, var2, color);
    error += Δerror;
    if (abs(error) >= 1f) { static if (countUp) { var2++; error -= 1f; } else { var2--; error += 1f; }}
  }
}

T max(T)(T a, T b) { return a>b?a:b; }
T min(T)(T a, T b) { return a<b?a:b; }

void line(T...)(int x0, int y0, int x1, int y1, T p) {
  execParams(p);
  static if (select!(back_rgb, T)!=-1) {
    SDL_LockSurface(display);
    scope(exit) { SDL_UnlockSurface(display); SDL_Flip(display); }
    auto xend=max(x0, x1);
    for (int x=min(x0, x1); x<=xend; ++x) {
      auto yend=max(y0, y1);
      for (int y=min(y0, y1); y<=yend; ++y) {
        putpixel(display, x, y, back);
      }
    }
  }
  static if (select!(box_rgb, T)!=-1) {
    line(x0, y0, x1, y0);
    line(x1, y0, x1, y1);
    line(x1, y1, x0, y1);
    line(x0, y1, x0, y0);
  }
  static if (select!(box_rgb, T)+select!(back_rgb, T)==-2) {
    SDL_LockSurface(display);
    scope(exit) { SDL_UnlockSurface(display); SDL_Flip(display); }
    bool steep = abs(y1 - y0) > abs(x1 - x0);
    void turn() { swap(x0, x1); swap(y0, y1); }
    if (steep) { if (y1 < y0) turn; }
    else { if (x1 < x0) turn; }
    bool stepUp=steep ? (x0 < x1) : (y0 < y1);
    if (steep) {
      if (stepUp) bresenham!(true, true)(x0, y0, x1, y1);
      else bresenham!(false, true)(x0, y0, x1, y1);
    } else {
      if (stepUp) bresenham!(true, false)(x0, y0, x1, y1);
      else bresenham!(false, false)(x0, y0, x1, y1);
    }
  }
}

import llvm.intrinsic;
alias llvm_sqrt sqrt;

template circle_bresenham_pass(bool first) {
  const string xy=(first?"x":"y");
  const string yx=(first?"y":"x");
  const string str="
    auto x="~(first?"xradius":"0")~";
    auto y="~(first?"0":"yradius")~";
    auto xchange=yradius*yradius*"~(first?"(1-2*xradius)":"1")~";
    auto ychange=xradius*xradius*"~(first?"1":"(1-2*yradius)")~";
    auto error=0;
    auto stopx="~(first?"y2square*xradius":"0")~";
    auto stopy="~(first?"0":"x2square*yradius")~";
    while (stopx"~(first?">=":"<=")~"stopy) {
      putpixel(display, cx+x, cy+y, color);
      putpixel(display, cx+x, cy-y, color);
      putpixel(display, cx-x, cy+y, color);
      putpixel(display, cx-x, cy-y, color);
      "~yx~"++;
      stop"~yx~"+="~xy~"2square;
      error+="~yx~"change;
      "~yx~"change+="~xy~"2square;
      if ((2*error+"~xy~"change)>0) {
        --"~xy~";
        stop"~xy~"-="~yx~"2square;
        error+="~xy~"change;
        "~xy~"change+="~yx~"2square;
      }
    }
  ";
}

void circle(T...)(int cx, int cy, int xradius, T t) {
  SDL_LockSurface(display);
  scope(exit) { SDL_UnlockSurface(display); SDL_Flip(display); }
  execParams(t);
  auto yradius=xradius;
  static if (T.length && is(T[0]: int)) yradius=t[0];
  static if (select!(back_rgb, T) != -1) {
    auto ratio=xradius*1f/yradius;
    for (int i=0; i<=yradius; ++i) {
      ushort j=cast(ushort)(sqrt(cast(real)(yradius*yradius-i*i))*ratio);
      for (int lx=cx-j; lx<=cx+j; ++lx) putpixel(display, lx, cy+i, back);
      for (int lx=cx-j; lx<=cx+j; ++lx) putpixel(display, lx, cy-i, back);
    }
  }
  auto x2square=2*xradius*xradius;
  auto y2square=2*yradius*yradius;
  { mixin(circle_bresenham_pass!(true).str); }
  { mixin(circle_bresenham_pass!(false).str); }
}

struct floodfill_node {
  int x, y;
  static floodfill_node opCall(int x, int y) {
    floodfill_node res;
    res.x=x; res.y=y;
    return res;
  }
}

void paint(T...)(int x, int y, T t) {
  SDL_LockSurface(display);
  scope(exit) { SDL_UnlockSurface(display); SDL_Flip(display); }
  execParams(t);
  bool border=true;
  if (select!(back_rgb, T) == -1) {
    back=pget(x, y);
    border=false;
  }
  bool check(rgb r) {
    if (border) return (r != back) && (r != color);
    else return r == back;
  }
  if (back == color) throw new Exception("Having identical backgrounds and foregrounds will severely mess up floodfill.");
  alias floodfill_node node;
  node[] queue;
  queue ~= node(x, y);
  size_t count=0;
  while (count<queue.length) {
    scope(exit) count++;
    with (queue[count]) {
      if (check(pget(x, y))) {
        int w=x, e=x;
        if (w<display.w) do w++; while ((w<display.w) && check(pget(w, y)));
        if (e>=0) do e--; while (e>=0 && check(pget(e, y)));
        //SDL_Flip(display);
        for (int i=e+1; i<w; ++i) {
          putpixel(display, i, y, color);
          if (y && check(pget(i, y-1)) && ((i==w-1)||!check(pget(i+1, y-1)))) queue ~= node(i, y-1);
          if ((y < display.h-1) && check(pget(i, y+1)) && ((i==w-1)||!check(pget(i+1, y+1)))) queue ~= node(i, y+1);
        }
      }
    }
  }
}

void screen(size_t w, size_t h) {
  display = SDL_SetVideoMode(w, h, 32, SDL_SWSURFACE);
}

void cls() { line(0, 0, display.w-1, display.h-1, Fill=Black); }
