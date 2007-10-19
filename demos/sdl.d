module sdl;

version(build) pragma(link,"SDL");

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
enum {
    SDL_GL_RED_SIZE,
    SDL_GL_GREEN_SIZE,
    SDL_GL_BLUE_SIZE,
    SDL_GL_ALPHA_SIZE,
    SDL_GL_BUFFER_SIZE,
    SDL_GL_DOUBLEBUFFER,
    SDL_GL_DEPTH_SIZE,
    SDL_GL_STENCIL_SIZE,
    SDL_GL_ACCUM_RED_SIZE,
    SDL_GL_ACCUM_GREEN_SIZE,
    SDL_GL_ACCUM_BLUE_SIZE,
    SDL_GL_ACCUM_ALPHA_SIZE,
    SDL_GL_STEREO,
    SDL_GL_MULTISAMPLEBUFFERS,
    SDL_GL_MULTISAMPLESAMPLES,
    SDL_GL_ACCELERATED_VISUAL,
    SDL_GL_SWAP_CONTROL
}
int SDL_GL_LoadLibrary(char*);
void* SDL_GL_GetProcAddress(char*);
int SDL_GL_SetAttribute(int,int);
int SDL_GL_GetAttribute(int,int*);
void SDL_GL_SwapBuffers();
void SDL_GL_UpdateRects(int,SDL_Rect*);
void SDL_GL_Lock();
void SDL_GL_Unlock();
enum : uint {
    SDL_INIT_TIMER=0x00000001,
    SDL_INIT_AUDIO=0x00000010,
    SDL_INIT_VIDEO=0x00000020,
    SDL_INIT_CDROM=0x00000100,
    SDL_INIT_JOYSTICK=0x00000200,
    SDL_INIT_NOPARACHUTE=0x00100000,
    SDL_INIT_EVENTTHREAD=0x00200000,
    SDL_INIT_EVERYTHING=0x0000FFFF
}

int SDL_Init(uint);
int SDL_InitSubSystem(uint);
int SDL_QuitSubSystem(uint);
int SDL_WasInit(uint);
void SDL_Quit();
