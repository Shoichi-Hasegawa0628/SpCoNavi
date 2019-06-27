COMPILE=g++
BUILD=build
PROGRAM=$(BUILD)/map_visualize
PROGRAM2=$(BUILD)/position_estimate
all:$(PROGRAM) $(PROGRAM2)

$(PROGRAM):$(BUILD)/map_visualize.o  $(BUILD)/position_visualize.o $(BUILD)/map_visualize_no_angle.o $(BUILD)/mathmatical.o $(BUILD)/foperation.o $(BUILD)/plestimate.o $(BUILD)/position_estimate.o
	g++ -o $(BUILD)/map_visualize  $(BUILD)/mathmatical.o $(BUILD)/map_visualize.o $(BUILD)/foperation.o $(BUILD)/plestimate.o `pkg-config --cflag opencv` `pkg-config --libs opencv`
	g++ -o $(BUILD)/map_visualize_no_angle  $(BUILD)/mathmatical.o $(BUILD)/map_visualize_no_angle.o $(BUILD)/foperation.o $(BUILD)/plestimate.o `pkg-config --cflag opencv` `pkg-config --libs opencv`
	g++ -o $(BUILD)/position_estimate $(BUILD)/mathmatical.o $(BUILD)/position_visualize.o $(BUILD)/foperation.o $(BUILD)/plestimate.o $(BUILD)/position_estimate.o `pkg-config --cflag opencv` `pkg-config --libs opencv`
$(BUILD)/map_visualize.o:src/map_visualize.cpp
	$(COMPILE) -c src/map_visualize.cpp -o $(BUILD)/map_visualize.o
$(BUILD)/map_visualize_no_angle.o:src/map_visualize_no_angle.cpp
	$(COMPILE) -c src/map_visualize_no_angle.cpp -o $(BUILD)/map_visualize_no_angle.o
$(BUILD)/position_visualize.o:src/position_visualize.cpp
	$(COMPILE) -c src/position_visualize.cpp -o $(BUILD)/position_visualize.o	
$(BUILD)/mathmatical.o:src/mathmatical.cpp
	$(COMPILE) -c src/mathmatical.cpp -o $(BUILD)/mathmatical.o
$(BUILD)/foperation.o:src/foperation.cpp
	$(COMPILE) -c src/foperation.cpp -o  $(BUILD)/foperation.o
$(BUILD)/plestimate.o:src/plestimate.cpp
	$(COMPILE) -c src/plestimate.cpp -o $(BUILD)/plestimate.o
$(BUILD)/position_estimate.o:src/position_estimate.cpp
	$(COMPILE) -c src/position_estimate.cpp -o $(BUILD)/position_estimate.o

.PHONY: clean
clean:
	rm -f $(BUILD)/*.o $(PROGRAM)
