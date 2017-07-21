CPPCOMP=pgc++
CPPFLAGS= -acc -ta=tesla:cc60,pinned -Minfo=all -fast -mp

LD=$(CPPCOMP)
LFLAGS=$(CPPFLAGS)

.PHONY: clean

run:  mandelbrot.o
	$(LD) mandelbrot.o -o run $(LFLAGS)

mandelbrot.o: mandelbrot.cpp
	$(CPPCOMP) $(CPPFLAGS) -c mandelbrot.cpp

clean:
	rm -f *.o run

cleanall:
	rm -f *.o run *.pgm
