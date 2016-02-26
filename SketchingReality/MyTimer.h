#pragma once

#include <string>
#include <map>

using namespace std;

class MyTimer {
private:
	bool valid;
	map<string, time_t> start_time;
	map<string, double> elapsed;
	map<string, FILE*> fp;

public:
	MyTimer();
	~MyTimer();

	void validate(bool valid = true);
	void start(const string& name);
	void end(const string& name);
};
