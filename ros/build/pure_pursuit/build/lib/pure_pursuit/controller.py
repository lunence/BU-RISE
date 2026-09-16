import math
import numpy as np

from pure_pursuit.utils import pt_to_pt_distance, sgn


"""
credit to purdue sigbots for pure pursuit function
"""

def clamp(n, smallest, largest):
    if n<smallest:
        return smallest
    elif n>largest:
        return largest
    else:
        return n

class PurePursuit:
    def __init__(self, lookAheadDis):
        self.last_found_index = 0
        self.lookAheadDis = lookAheadDis

        self.reached_target = False
        self.exit = False

        self.success_radius = 0.15      # TODO: change back to 15 cm
        self.stall_radius = 0.20        # 30 cm
        self.stall_time = 15.0           # seconds
        self.stall_counter = 0
        self.dt = 0.05                  # controller period

        # stopping prediction
        self.stop_prediction_time = 3.0   # seconds into future
        self.last_speed = 0.0

        self.last_turn_direction = 1

    def predict_stop_position(self, currentPos, currentHeading, velocity):
        """
        Predict where the robot will be after continuing for
        stop_prediction_time seconds.
        """

        t = self.stop_prediction_time

        x = (
            currentPos[0]
            + velocity * math.sin(math.radians(currentHeading)) * t
        )

        y = (
            currentPos[1]
            + velocity * math.cos(math.radians(currentHeading)) * t
        )

        return np.array([x, y])

    def compute_control(self, path, currentPos, currentHeading):
        # initialize proportional controller constant
        Kp_lin = 0.1
        Kp_turn = 0.25

        # limo max speed 
        max_linear = 0.25
        max_angular = 0.5 #0.5

        if self.reached_target or self.exit:
            return 0, 0
        else:
            # for path length 1
            if len(path) == 1:
                goalPt = path[0]

            # for paths longer than 1 point

            # # extract currentX and currentY
            # currentX = currentPos[0]
            # currentY = currentPos[1]

            # # use for loop to search intersections
            # lastFoundIndex = self.last_found_index
            # intersectFound = False
            # startingIndex = lastFoundIndex

            # for i in range (startingIndex, len(path)-1):

            #     # beginning of line-circle intersection code
            #     x1 = path[i][0] - currentX
            #     y1 = path[i][1] - currentY
            #     x2 = path[i+1][0] - currentX
            #     y2 = path[i+1][1] - currentY
            #     dx = x2 - x1
            #     dy = y2 - y1
            #     dr = math.sqrt (dx**2 + dy**2)
            #     D = x1*y2 - x2*y1
            #     discriminant = (self.lookAheadDis**2) * (dr**2) - D**2

            #     if discriminant >= 0:
            #         sol_x1 = (D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
            #         sol_x2 = (D * dy - sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
            #         sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
            #         sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2

            #         sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
            #         sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]
            #         # end of line-circle intersection code

            #         minX = min(path[i][0], path[i+1][0])
            #         minY = min(path[i][1], path[i+1][1])
            #         maxX = max(path[i][0], path[i+1][0])
            #         maxY = max(path[i][1], path[i+1][1])

            #         # if one or both of the solutions are in range
            #         if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) or ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):

            #             foundIntersection = True

            #             # if both solutions are in range, check which one is better
            #             if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) and ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):
            #                 # make the decision by compare the distance between the intersections and the next point in path
            #                 if pt_to_pt_distance(sol_pt1, path[i+1]) < pt_to_pt_distance(sol_pt2, path[i+1]):
            #                     goalPt = sol_pt1
            #                 else:
            #                     goalPt = sol_pt2
                        
            #             # if not both solutions are in range, take the one that's in range
            #             else:
            #                 # if solution pt1 is in range, set that as goal point
            #                 if (minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY):
            #                     goalPt = sol_pt1
            #                 else:
            #                     goalPt = sol_pt2
                        
            #             # only exit loop if the solution pt found is closer to the next pt in path than the current pos
            #             if pt_to_pt_distance (goalPt, path[i+1]) < pt_to_pt_distance ([currentX, currentY], path[i+1]):
            #                 # update lastFoundIndex and exit
            #                 lastFoundIndex = i
            #                 break
            #             else:
            #                 # in case for some reason the robot cannot find intersection in the next path segment, but we also don't want it to go backward
            #                 lastFoundIndex = i+1
                        
            #         # if no solutions are in range
            #         else:
            #             foundIntersection = False
            #             # no new intersection found, potentially deviated from the path
            #             # follow path[lastFoundIndex]
            #             goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]

            # # update last found index variable
            # self.last_found_index = lastFoundIndex

            # normal distance
            linearError = np.sqrt(
                (goalPt[1]-currentPos[1])**2 +
                (goalPt[0]-currentPos[0])**2
            )


            # predict where robot will be before it stops
            predicted_stop_pos = self.predict_stop_position(
                currentPos,
                currentHeading,
                self.last_speed
            )


            stop_error = np.sqrt(
                (goalPt[1]-predicted_stop_pos[1])**2 +
                (goalPt[0]-predicted_stop_pos[0])**2
            )

            # obtained goal point, now compute turn vel
            # calculate absTargetAngle with the atan2 function
            absTargetAngle = math.atan2(
                goalPt[0]-currentPos[0],
                goalPt[1]-currentPos[1]
            ) *180/math.pi 
            if absTargetAngle < 0: absTargetAngle += 360

            #
            # compute turn error by finding the minimum angle
            turnError = (absTargetAngle - currentHeading + 180) % 360 - 180

            # # Prevent sign flip around 180°
            # if abs(abs(turnError) - 180) < 10:
            #     turnError = 180 * self.last_turn_direction
            # else:
            #     self.last_turn_direction = 1 if turnError > 0 else -1

            turnError_rad = math.radians(turnError)
            # apply proportional controller
            # TODO: WAYYY too high for a limo
            turnVel = -Kp_turn*turnError_rad
            turnVel = clamp(turnVel, -max_angular, max_angular)

            # print("linear error:", linearError)
            # print("stop error:", stop_error)

            print("Target angle:", absTargetAngle)
            # print("turn error:", turnError)

            # calculate speed - slow down in sharp turns
            if abs(turnError) > 45:
                linearVel = 0
            else:
                linearVel = Kp_lin * linearError
                linearVel = linearVel / (1 + abs(turnVel))
                linearVel = clamp(linearVel, 0, max_linear)

            # TODO: if more than one point in the path, add condition that checks if it is last point in path
            # Reached target normally
            if(linearError<self.success_radius):
                self.reached_target = True

            # Within larger "good enough" region
            elif linearError < self.stall_radius:
                self.stall_counter += 1

                if self.stall_counter * self.dt >= self.stall_time:
                    print("Goal timeout reached.")
                    self.exit = True

            # Left the region, reset timer
            else:
                self.stall_counter = 0

        self.last_speed = linearVel

        return linearVel, turnVel
