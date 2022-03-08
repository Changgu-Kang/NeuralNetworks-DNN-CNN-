#include "NeuralNetwork.h"

size_t FindFiles(std::string strPath, std::vector<std::string> &q, bool bIncludeSub)
{
	if (strPath.empty()) return 0;

	if (strPath.at(strPath.length() - 1) != '\\')
		strPath.push_back('\\');

	std::string strFindPattern = strPath + "*.*";

	WIN32_FIND_DATA     fd;
	auto    hFind = FindFirstFile(strFindPattern.c_str(), &fd);
	if (INVALID_HANDLE_VALUE == hFind)
		return 0;

	size_t nFind = 0;
	do {
		if ((fd.dwFileAttributes & FILE_ATTRIBUTE_SYSTEM) != 0)
			continue;
		std::string strFound = strPath + fd.cFileName;
		if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
			if (bIncludeSub && strcmp(fd.cFileName, ".") && strcmp(fd.cFileName, ".."))
				nFind += FindFiles(strFound, q, true);
		}
		else
			q.push_back(strFound);
	} while (FindNextFile(hFind, &fd));
	FindClose(hFind);
	return nFind;
}