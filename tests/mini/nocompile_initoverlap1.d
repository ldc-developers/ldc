struct Vector {
        union { float x; float y; }
        const static Vector zero = { x : 0, y : 0 };
}

struct HBoxLayout {
        Vector padding  = Vector.zero;
}
