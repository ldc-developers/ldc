// https://github.com/ldc-developers/ldc/issues/3556
// RUN: %ldc -run %s

class C {
    int counter = 1;

    void test1() {
        assert(counter == 1);
        ++counter;
    }

    void run1() {
        class C2 {
            int counter2 = 11;

            class C3 {
                void run3() {
                    test1();
                    test2();
                    ++counter;
                    ++counter2;
                }
            }

            void test2() {
                assert(counter == 2);
                ++counter;
                assert(counter2 == 11);
                ++counter2;
            }

            void run2() {
                auto c3 = new C3;
                c3.run3();
                ++counter;
                ++counter2;
            }
        }

        auto c2 = new C2;
        c2.run2();
        assert(c2.counter2 == 14);
        ++counter;
    }
}

void main() {
    auto c = new C;
    c.run1();
    assert(c.counter == 6);
}
