#if !defined(COCO_STRINGBUILDER_H__)
#define COCO_STRINGBUILDER_H__

#include<stddef.h>

namespace Coco {

class StringBuilder
{
public:
	StringBuilder(int capacity = 32);
	StringBuilder(char const *val);
	
	virtual ~StringBuilder();
	void Append(char val);
	void Append(char const *val);
	char* ToString();
	size_t GetLength() { return length; };

private:
	void Init(int capacity);
	char *data;
	size_t capacity;
	size_t length;
};

}; // namespace

#endif // !defined(COCO_STRINGBUILDER_H__)
