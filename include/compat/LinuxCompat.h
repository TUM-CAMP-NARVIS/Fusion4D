#ifndef PEABODY_LINUX_COMPAT_H
#define PEABODY_LINUX_COMPAT_H

#ifndef _WIN32

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <thread>

using WORD = unsigned short;
using DWORD = unsigned long;
using HANDLE = void *;
using LPVOID = void *;
using HRESULT = long;
using WCHAR = wchar_t;

struct LARGE_INTEGER
{
	long long QuadPart;
};

#ifndef S_OK
#define S_OK 0
#endif

#endif
#ifndef E_FAIL
#define E_FAIL 0x80004005L
#endif
#ifndef __forceinline
#define __forceinline inline __attribute__((always_inline))
#endif

#define FOREGROUND_BLUE 0x0001
#define FOREGROUND_GREEN 0x0002
#define FOREGROUND_RED 0x0004
#define FOREGROUND_INTENSITY 0x0008
#define BACKGROUND_BLUE 0x0010
#define BACKGROUND_GREEN 0x0020
#define BACKGROUND_RED 0x0040
#define BACKGROUND_INTENSITY 0x0080
#define STD_OUTPUT_HANDLE ((DWORD) - 11)
#define STD_ERROR_HANDLE ((DWORD) - 12)

inline HANDLE GetStdHandle(DWORD) { return nullptr; }
inline bool SetConsoleTextAttribute(HANDLE, WORD) { return true; }
inline HANDLE GetCurrentProcess() { return nullptr; }
inline bool TerminateProcess(HANDLE, unsigned int exit_code)
{
	std::exit(static_cast<int>(exit_code));
}

inline bool QueryPerformanceCounter(LARGE_INTEGER *value)
{
	const auto now = std::chrono::steady_clock::now().time_since_epoch();
	value->QuadPart = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
	return true;
}

inline bool QueryPerformanceFrequency(LARGE_INTEGER *value)
{
	value->QuadPart = 1000000000LL;
	return true;
}

inline void Sleep(DWORD milliseconds)
{
	std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

inline int GetPrivateProfileString(
	const WCHAR *,
	const WCHAR *,
	const WCHAR *default_value,
	WCHAR *returned_string,
	DWORD size,
	const WCHAR *)
{
	if (returned_string == nullptr || size == 0)
		return 0;
	const WCHAR *value = default_value != nullptr ? default_value : L"";
	std::wcsncpy(returned_string, value, size - 1);
	returned_string[size - 1] = L'\0';
	return static_cast<int>(std::wcslen(returned_string));
}

inline int WritePrivateProfileString(const WCHAR *, const WCHAR *, const WCHAR *, const WCHAR *)
{
	return 0;
}

inline int fopen_s(FILE **file, const char *filename, const char *mode)
{
	*file = std::fopen(filename, mode);
	return *file == nullptr;
}

#ifndef _isnan
#define _isnan std::isnan
#endif

#ifndef sprintf_s
#define sprintf_s std::snprintf
#endif

#ifndef fprintf_s
#define fprintf_s std::fprintf
#endif

#ifndef fscanf_s
#define fscanf_s std::fscanf
#endif

#ifndef vcl_endl
#define vcl_endl std::endl
#endif

inline char *_strrev(char *str)
{
	if (str == nullptr)
		return nullptr;
	char *left = str;
	char *right = str + std::strlen(str);
	while (left < right && left < --right)
	{
		const char tmp = *left;
		*left++ = *right;
		*right = tmp;
	}
	return str;
}

#endif
