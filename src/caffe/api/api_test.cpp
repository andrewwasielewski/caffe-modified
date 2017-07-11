#include "CaffeAPI.h"
#include <time.h>
#include <iostream>

#include <windows.h>
#include <psapi.h>
#include <process.h>
#pragma comment( lib, "psapi.lib" )

void GetMemoryInfo() {

	HANDLE hProcess;
	PROCESS_MEMORY_COUNTERS pmc;

	// Print the process identifier.

	int processID = _getpid();

	printf("\nProcess ID: %u\n", processID);

	// Print information about the memory usage of the process.

	hProcess = OpenProcess( PROCESS_QUERY_INFORMATION |
	PROCESS_VM_READ,
	FALSE, processID);
	if (NULL == hProcess) {
		printf("hProcess is NULL\n");
		return;
	}

	if (GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc))) {
		printf("\tPeakWorkingSetSize: %f\n",
				pmc.PeakWorkingSetSize * 1.0 / 1024 / 1024);
		printf("\tWorkingSetSize: %f\n",
				pmc.WorkingSetSize * 1.0 / 1024 / 1024);
	}
	CloseHandle(hProcess);
}
int main(int argc, char **argv) {
	std::cout<<"***********Initial memory profile**********";
	GetMemoryInfo();
	Caffe_API caffeapi;
	int mode = atoi(argv[1]);
	caffeapi.setMode(mode);
	std::cout<<"**********Memory profile after setting mode********";
	GetMemoryInfo();
	caffeapi.readNetwork(argv[2],argv[3]);
	std::cout<<"**********Memory profile after reading network********";
	GetMemoryInfo();
	caffeapi.readTestDataFromBinFile("C:\\Temp\\aorta_testimage.bin","data");
	clock_t begin = clock();
	std::cout<<"**********Memory profile after reading data***********";
	GetMemoryInfo();
	caffeapi.run();
	std::cout<<"**********Memory profile after inference*********";
	GetMemoryInfo();
	caffeapi.resetNet();

	std::cout<<"**********Memory profile after cleaning up the memory********";

	GetMemoryInfo();
	std::cout<<"Time taken to run the segmentation:"<<double(clock()-begin)/1000.0<<" seconds\n";
	return 0;
}
