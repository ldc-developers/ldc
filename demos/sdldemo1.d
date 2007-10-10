module sdldemo1;
import sdl;
void main()
{
    auto disp = SDL_SetVideoMode(640,480,0,SDL_HWSURFACE|SDL_DOUBLEBUF);
    auto r = SDL_Rect(0,190,100,100);
    auto c = SDL_MapRGBA(disp.format,255,100,0,255);
    while (r.x < disp.w-100) {
        SDL_FillRect(disp, null, 0);
        SDL_FillRect(disp, &r, c);
        SDL_Flip(disp);
        r.x++;
    }
}

