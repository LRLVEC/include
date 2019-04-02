#pragma once
#include <_File.h>

//Only support 24bit bmp!
struct BMP
{
#pragma pack(1)
	struct Header
	{
		char identifier[2];
		unsigned int fileSize;
		unsigned int reserved;
		unsigned int dataOffset;
		unsigned int headerSize;
		unsigned int width;
		unsigned int height;
		unsigned short planeNum;
		unsigned short bitsPerPixel;
		unsigned int compressionMethod;
		unsigned int dataSize;
		unsigned int verticalPixelsPerMeter;
		unsigned int horizontalPixelsPerMeter;
		unsigned int colorNum;
		unsigned int importantColorNum;
		void printInfo()const
		{
			::printf("BMP file info:\n");
			::printf("\tFile Size:\t%u\n", fileSize);
			::printf("\tWidth:\t\t%u\n", width);
			::printf("\tHeight:\t\t%u\n", height);
		}
	};
	struct Pixel
	{
		unsigned char b;
		unsigned char g;
		unsigned char r;
	};
#pragma pack()

	Header header;
	Pixel* data;

	BMP()
		:
		header(),
		data(nullptr)
	{
	}
	BMP(String<char>const& _path)
	{
		FILE* temp(::fopen(_path.data, "rb+"));
		::fseek(temp, 0, SEEK_SET);
		::fread(&header, 1, 54, temp);
		::fseek(temp, header.dataOffset, SEEK_SET);
		if (header.width % 4)
			data = (BMP::Pixel*)::malloc(3u * header.width * header.height + 4);
		else
			data = (BMP::Pixel*)::malloc(3u * header.width * header.height);;
		for (int c0(0); c0 < header.height; ++c0)
			::fread((data + header.width * c0), 4, header.width * 3/4, temp);
		::fclose(temp);
	}
	~BMP()
	{
		::free(data);
	}

	bool checkType()const
	{
		return header.identifier[0] == 'B' && header.identifier[1] == 'M';
	}
	void printInfo()const
	{
		header.printInfo();
	}
};

//File...
inline BMP File::readBMP()const
{
	if (!this)return BMP();
	BMP r;
	FILE* temp(::fopen((property.path + property.file.name).data, "rb+"));
	::fseek(temp, 0, SEEK_SET);
	::fread(&r.header, 1, 54, temp);
	::fseek(temp, r.header.dataOffset, SEEK_SET);
	if (r.header.width % 4)
		r.data = (BMP::Pixel*)::malloc(3u * r.header.width * r.header.height + 4);
	else
		r.data = (BMP::Pixel*)::malloc(3u * r.header.width * r.header.height);
	for (int c0(0); c0 < r.header.height; ++c0)
		::fread((r.data + r.header.width), 4, r.header.width, temp);
	::fclose(temp);
	return r;
}
inline BMP File::readBMP(String<char> const& _name)const
{
	if (!this)return BMP();
	BMP r;
	FILE* temp(::fopen((property.path + _name).data, "rb+"));
	::fseek(temp, 0, SEEK_SET);
	::fread(&r.header, 1, 54, temp);
	::fseek(temp, r.header.dataOffset, SEEK_SET);
	if (r.header.width % 4)
		r.data = (BMP::Pixel*)::malloc(3u * r.header.width * r.header.height + 4);
	else
		r.data = (BMP::Pixel*)::malloc(3u * r.header.width * r.header.height);
	for (int c0(0); c0 < r.header.height; ++c0)
		::fread((r.data + r.header.width), 4, r.header.width, temp);
	::fclose(temp);
	return r;
}