module qd;

alias char[] string;

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
  SDL_Surface *SDL_CreateRGBSurface(uint flags, int width, int height, int depth, uint Rmask=0, uint Gmask=0, uint Bmask=0, uint Amask=0);
  int SDL_Flip(SDL_Surface *);
  void SDL_UpdateRect (SDL_Surface *screen, int x, int y, uint w, uint h);
  int SDL_UpperBlit(SDL_Surface *src, SDL_Rect *srcrect, SDL_Surface *dst, SDL_Rect *dstrect);
  alias SDL_UpperBlit SDL_BlitSurface;
  int SDL_SetAlpha(SDL_Surface *surface, uint flags, ubyte alpha);
  int SDL_SetColorKey(SDL_Surface *surface, uint flag, uint key);
  int SDL_FillRect(SDL_Surface *dst, SDL_Rect *dstrect, uint color);
  const uint SDL_SWSURFACE=0;
  const uint SDL_HWSURFACE=1;
  const uint SDL_DOUBLEBUF=0x40000000;
  const uint SDL_FULLSCREEN=0x80000000;
  const uint SDL_SRCALPHA=0x00010000;
  const uint SDL_SRCCOLORKEY=0x00001000;
  void SDL_Delay(uint ms);
  uint SDL_GetTicks();

  enum SDLKey {
    Unknown = 0, First = 0, 
    Escape = 27,
    LCtrl = 306,
  }
  enum SDLMod {
    KMOD_NONE  = 0x0000,
    KMOD_LSHIFT= 0x0001, KMOD_RSHIFT= 0x0002,
    KMOD_LCTRL = 0x0040, KMOD_RCTRL = 0x0080, KMOD_CTRL  = 0x00C0,
    KMOD_LALT  = 0x0100, KMOD_RALT  = 0x0200, KMOD_ALT   = 0x0300,
    KMOD_LMETA = 0x0400, KMOD_RMETA = 0x0800,
    KMOD_NUM   = 0x1000, KMOD_CAPS  = 0x2000, KMOD_MODE  = 0x4000,
    KMOD_RESERVED = 0x8000
  };

  struct SDL_keysym { ubyte scancode; SDLKey sym; SDLMod mod; ushort unicode; }
  enum SDL_EventType : ubyte {
    NoEvent=0, Active, KeyDown, KeyUp,
    MouseMotion, MouseButtonDown, MouseButtonUp,
    JoyAxisMotion, JoyBallMotion, JoyHatMotion, JoyButtonDown, JoyButtonUp,
    Quit, SysWMEvent
  }
  union SDL_Event {
    SDL_EventType type;
    struct Active { SDL_EventType type, gain, state; }; Active active;
    struct Key { SDL_EventType type, which, state; SDL_keysym keysym; }; Key key;
    struct Motion { SDL_EventType type, which, state; ushort x, y; short xrel, yrel; }; Motion motion;
    struct Button { SDL_EventType type, which, button, state; ushort x, y; }; Button button;
    struct Jaxis { SDL_EventType type, which, axis; short value; }; Jaxis jaxis;
    struct Jball { SDL_EventType type, which, ball; short xrel, yrel; }; Jball jball;
    struct Jhat { SDL_EventType type, which, hat, value; }; Jhat jhat;
    struct Jbutton { SDL_EventType type, which, button, state; }; Jbutton jbutton;
    struct Resize { SDL_EventType type; int w, h; }; Resize resize;
    struct Expose { SDL_EventType type; }; Expose expose;
    struct Quit { SDL_EventType type; }; Quit quit;
    struct User { SDL_EventType type; int code; void *data1, data2; }; User user;
    struct Syswm { SDL_EventType type; void *msg; }; Syswm syswm;
  }

  int SDL_PollEvent(SDL_Event *event);
}

SDL_Surface *display;

void putpixel32(int x, int y, ubyte[4] col) {
  uint *bufp = cast(uint *)display.pixels + y*display.pitch/4 + x;
  *bufp = SDL_MapRGBA(display.format, col[0], col[1], col[2], col[3]);
}

void putpixel32(int x, int y, ubyte[3] col) {
  uint *bufp = cast(uint *)display.pixels + y*display.pitch/4 + x;
  *bufp = SDL_MapRGBA(display.format, col[0], col[1], col[2], 0);
}

void getpixel32(int x, int y, ubyte[4] *col) {
  uint *bufp = cast(uint *)display.pixels + y*display.pitch/4 + x;
  SDL_GetRGBA(*bufp, display.format, &(*col)[0], &(*col)[1], &(*col)[2], &(*col)[3]);
}

align(1)
struct rgb {
  ubyte[3] values;
  ubyte r() { return values[0]; }
  ubyte g() { return values[1]; }
  ubyte b() { return values[2]; }
  rgb opCat(rgb other) {
    rgb res;
    foreach (id, ref v; res.values) v=cast(ubyte)((values[id]+other.values[id])/2);
    return res;
  }
  bool opEquals(rgb r) {
    return values == r.values;
  }
}

void putpixel(int x, int y, ubyte[4] col) {
  if ( (x<0) || (y<0) || (x!<display.w) || (y!<display.h) ) return;
  putpixel32(x, y, col);
}

void hline(int x, int y, int w, rgb r) {
  hline(x, y, w, SDL_MapRGBA(display.format, r.values[0], r.values[1], r.values[2], 0));
}
void hline(int x, int y, int w, uint c) {
  if ( (y<0) || (y!<display.h) ) return;
  if (x<0) { w+=x; x=0; }
  if (w<0) return;
  if ( (x+w) !<display.w) w=display.w-x-1;
  auto cur = cast(uint *)display.pixels + y*display.pitch/4 + x;
  foreach (ref value; cur[0..w+1]) value=c;
}

const rgb White={[255, 255, 255]};
const rgb Black={[0, 0, 0]};
const rgb Red={[255, 0, 0]};
const rgb Green={[0, 255, 0]};
const rgb Blue={[0, 0, 255]};
const rgb Yellow={[255, 255, 0]};
const rgb Cyan={[0, 255, 255]};
const rgb Purple={[255, 0, 255]};
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

bool doFlip=true;
void flip() { SDL_Flip(display); }
void flip(bool target) { doFlip=target; }
scope class groupDraws {
  bool wasOn;
  this() { wasOn=doFlip; flip=false; }
  ~this() { if (wasOn) { flip=true; flip; } }
}

void execParams(T...)(T params) {
  const int bcol=select!(back_rgb, T);
  static if (bcol != -1) back=cast(rgb) params[bcol];
  const int col=select!(rgb, T);
  static if (col != -1) color=params[col];
  else static if (bcol != -1) color=back;
  const int boxcol=select!(box_rgb, T);
  static if (boxcol != -1) color=cast(rgb) params[boxcol];
}

void tintfill(int x1, int y1, int x2, int y2, rgb color) {
  SDL_LockSurface(display);
  scope(exit) { SDL_UnlockSurface(display); if (doFlip) flip; }
  ubyte[4] c;
  for (int x=x1; x<x2; ++x) {
    for (int y=y1; y<y2; ++y) {
      getpixel32(x, y, &c);
      c[0]=cast(ubyte)(c[0]*178+color.r*77)>>8;
      c[1]=cast(ubyte)(c[1]*178+color.g*77)>>8;
      c[2]=cast(ubyte)(c[2]*178+color.b*77)>>8;
      putpixel32(x, y, c);
    }
  }
}

void pset(T...)(int x, int y, T params) {
  SDL_LockSurface(display);
  scope(exit) { SDL_UnlockSurface(display); if (doFlip) flip; }
  execParams(params);
  putpixel32(x, y, color.values);
}

rgb pget(int x, int y) {
  SDL_LockSurface(display);
  scope(exit) SDL_UnlockSurface(display);
  ubyte[4] c;
  getpixel32(x, y, &c);
  rgb res; res.values[]=c[0..3]; return res;
}

void swap(T)(ref T a, ref T b) { T c=a; a=b; b=c; }
T abs(T)(T a) { return (a<0) ? -a : a; }

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
  ubyte[4] col; col[0..3]=color.values;
  for (auto var1 = mixin(name~'0'); var1 <= mixin(name~'1'); ++var1) {
    static if (steep) putpixel(var2, var1, col);
    else putpixel(var1, var2, col);
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
    scope(exit) { SDL_UnlockSurface(display); if (doFlip) flip; }
    auto yend=max(y0, y1);
    for (int y=min(y0, y1); y<=yend; ++y) {
      hline(min(x0, x1), y, max(x0, x1)-min(x0, x1), back);
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
    scope(exit) { SDL_UnlockSurface(display); if (doFlip) flip; }
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

import llvmdc.intrinsics;
alias llvm_sqrt_f32 sqrt;
alias llvm_sqrt_f64 sqrt;
version(X86)
{
    alias llvm_sqrt_f80 sqrt;
}
else
{
    static import tango.stdc.math;
    real sqrt(real x)
    {
        return tango.stdc.math.sqrtl(x);
    }
}


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
      putpixel(cx+x, cy+y, col);
      putpixel(cx+x, cy-y, col);
      putpixel(cx-x, cy+y, col);
      putpixel(cx-x, cy-y, col);
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

void circle(T...)(T t) {
  static assert(T.length!<3, "Circle: Needs x, y and radius");
  int cx=t[0], cy=t[1], xradius=t[2];
  SDL_LockSurface(display);
  scope(exit) { SDL_UnlockSurface(display); if (doFlip) flip; }
  execParams(t[3..$]);
  auto yradius=xradius;
  if (xradius!>0) return;
  static if (T.length>3 && is(T[3]: int)) yradius=t[3];
  static if (select!(back_rgb, T) != -1) {
    auto ratio=xradius*1f/yradius;
    auto back_sdl=SDL_MapRGBA(display.format, back.values[0], back.values[1], back.values[2], 0);
    for (int i=0; i<=yradius; ++i) {
      ushort j=cast(ushort)(sqrt(cast(real)(yradius*yradius-i*i))*ratio);
      hline(cx-j, cy+i, 2*j, back_sdl);
      hline(cx-j, cy-i, 2*j, back_sdl);
    }
  }
  auto x2square=2*xradius*xradius;
  auto y2square=2*yradius*yradius;
  ubyte[4] col; col[0..3]=color.values;
  { mixin(circle_bresenham_pass!(true).str); }
  { mixin(circle_bresenham_pass!(false).str); }
}

float distance(float x1, float y1, float x2, float y2) {
  auto x=x1-x2, y=y1-y2;
  return sqrt(x*x+y*y);
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
  scope(exit) { SDL_UnlockSurface(display); if (doFlip) flip; }
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
          putpixel32(i, y, color.values);
          if (y && check(pget(i, y-1)) && ((i==w-1)||!check(pget(i+1, y-1)))) queue ~= node(i, y-1);
          if ((y < display.h-1) && check(pget(i, y+1)) && ((i==w-1)||!check(pget(i+1, y+1)))) queue ~= node(i, y+1);
        }
      }
    }
  }
}

struct screen {
  static {
    void opCall(size_t w, size_t h) {
      display = SDL_SetVideoMode(w, h, 32, SDL_SWSURFACE | SDL_DOUBLEBUF);
    }
    int width() { return display.w; }
    int height() { return display.h; }
  }
}


void cls(rgb fill=Black) { line(0, 0, display.w-1, display.h-1, Fill=fill); }

void events(void delegate(int, bool) key=null, void delegate(int, int, ubyte, bool) mouse=null) {
  SDL_Event evt;
  while (SDL_PollEvent(&evt)) {
    switch (evt.type) {
      case SDL_EventType.MouseMotion:
        with (evt.motion) if (mouse) mouse(x, y, 0, false);
        break;
      case SDL_EventType.MouseButtonDown:
        with (evt.button) if (mouse) mouse(x, y, button, true);
        break;
      case SDL_EventType.MouseButtonUp:
        with (evt.button) if (mouse) mouse(x, y, button, false);
        break;
      case SDL_EventType.KeyDown:
        if (key) key(evt.key.keysym.sym, true);
      case SDL_EventType.KeyUp:
        if (key) key(evt.key.keysym.sym, false);
        break;
      case SDL_EventType.Quit:
        throw new Exception("Quit");
        break;
      default: break;
    }
  }
}

void events(void delegate(int) key, void delegate(int, int, ubyte, bool) mouse=null) {
  events((int a, bool b) {
    if (b) key(a);
  }, mouse);
}

void events(void delegate(int) key, void delegate(int, int) mouse) {
  events(key, (int x, int y, ubyte b, bool p) { mouse(x, y); });
}

void events(void delegate(int, bool) key, void delegate(int, int) mouse) {
  events(key, (int x, int y, ubyte b, bool p) { mouse(x, y); });
}

void sleep(float secs)
{
    assert(secs >= 0);
    uint ms = cast(uint)(secs * 1000);
    SDL_Delay(ms);
}
