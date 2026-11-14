#include <iostream>
#include <string>

#include "kv_cache.h"
#include "zipf.h"

int main() {
    LRUCache cache(20);

    cache.Put("a", "11111");
    cache.Put("b", "22222");

    std::string value;

    if (cache.Get("a", value)) {
        std::cout << "Get a = " << value << std::endl;
    }

    cache.Put("c", "3333333333");

    std::cout << "Cache items: " << cache.ItemCount() << std::endl;
    std::cout << "Cache size bytes: " << cache.SizeBytes() << std::endl;
    std::cout << "Hit rate: " << cache.HitRate() << std::endl;

    ZipfGenerator zipf(10, 0.99);

    std::cout << "Zipf samples:" << std::endl;
    for (int i = 0; i < 20; ++i) {
        std::cout << zipf.Next() << " ";
    }
    std::cout << std::endl;

    return 0;
}
