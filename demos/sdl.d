module sdl;

version(build)
    pragma(link,"SDL");

extern(C):
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
int SDL_FillRect(SDL_Surface*,SDL_Rect*,uint);
enum : uint {
    SDL_SWSURFACE=0,
    SDL_HWSURFACE=1,
    SDL_DOUBLEBUF=0x40000000,
    SDL_FULLSCREEN=0x80000000
}

