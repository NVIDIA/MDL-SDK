#if !defined(COCO_STRINGBUILDER_H__)
#define COCO_STRINGBUILDER_H__

#include<stddef.h>

namespace Coco {

class StringBuilder  
{
public:
	StringBuilder(int capacity = 32);
	StringBuilder(const wchar_t *val);
	
	virtual ~StringBuilder();
	void Append(const wchar_t val);
	void Append(const wchar_t *val);
	wchar_t* ToString();
	int GetLength() { return length; };

private:
	void Init(int capacity);
	wchar_t *data;
	int capacity;
	int length;
};

}; // namespace

#endif // !defined(COCO_STRINGBUILDER_H__)
