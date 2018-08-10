all:
	g++ *.cpp -o Coco $(CFLAGS) 

clean:
	rm -f Coco

install:
	ln -s /usr/lib/coco-cpp/Coco $(DESTDIR)/usr/bin/cococpp
	install -m 0755 Coco $(DESTDIR)/usr/lib/coco-cpp
	install -m 0644 *frame $(DESTDIR)/usr/share/coco-cpp

