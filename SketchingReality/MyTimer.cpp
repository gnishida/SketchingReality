#include "MyTimer.h"
#include <time.h>

MyTimer::MyTimer() {
	valid = true;
}

MyTimer::~MyTimer() {
	if (valid) {
		for (auto it = fp.begin(); it != fp.end(); ++it) {
			fclose(it->second);
		}
	}
}

void MyTimer::validate(bool valid) {
	this->valid = valid;
}

void MyTimer::start(const string& name) {
	if (valid) {
		if (fp.find(name) == fp.end()) {
			char filename[256];
			sprintf(filename, "timer_%s.txt", name.c_str());
			fp[name] = fopen(filename, "w");
		}

		start_time[name] = clock();
	}
}

void MyTimer::end(const string& name) {
	if (valid) {
		if (start_time.find(name) == start_time.end()) {
			throw "Timer name not found.";
		}

		time_t t = clock();
		elapsed[name] += t - start_time[name];
		fprintf(fp[name], "%lf\n", elapsed[name]);
	}
}

