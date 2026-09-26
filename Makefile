CC ?= gcc
CPPFLAGS ?=
CFLAGS ?= -O3
CFLAGS += -fopenmp -Wall -Wextra -Wpedantic
LDFLAGS ?=
LDLIBS ?= -lgmp

TARGET ?= pi

.PHONY: all clean

all: $(TARGET)

$(TARGET): pi.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	$(RM) pi pi.exe
