ifneq ($(KERNELRELEASE),)
obj-m   := mutant.o
# mutant-objs := /protocol/tcp_pcc.o
else
KDIR    := /lib/modules/$(shell uname -r)/build 
ODIR    := $(shell pwd)/build
PWD     := $(shell pwd)
MY_CFLAGS += -g -DDEBUG
ccflags-y += -I/usr/src/linux-5.4.231/include

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

debug:
	$(MAKE) -C $(KDIR) M=$(PWD) modules EXTRA_CFLAGS="$(MY_CFLAGS)"

install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install

%:
	$(MAKE) -C $(KDIR) M=$(PWD) $@

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

endif

