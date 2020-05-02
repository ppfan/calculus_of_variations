"""
Solve Brachistochrone Curve by Tensorflow
"""


import tensorflow.compat.v1 as tf
import collections
import matplotlib.pyplot as plt
tf.disable_v2_behavior()
Point = collections.namedtuple('Point', ['x', 'y'])

VERBOSE = False
# Gravity
G = 9.8
PI = 3.1415926


class BrachistochroneCurve:
    """
    """
    def __init__(self, start_point, end_point):
        assert start_point.x < end_point.x and start_point.y > end_point.y

        self.dx = 0.05
        self.num_vars = int((end_point.x - start_point.x) / self.dx) + 1
        self.dy = (end_point.y - start_point.y) / (self.num_vars - 1)

        self.Points = []
        # Начальные точки
        x = tf.Variable(start_point.x, trainable=False, name='px_start')
        y = tf.Variable(start_point.y, trainable=False, name='py_start')
        self.Points.append([x, y])


        for i in range(1, self.num_vars - 1):
            x = tf.Variable(start_point.x + i * self.dx, trainable=False, name='px{}'.format(i))
            y = tf.Variable(start_point.y + i * self.dy, name='py{}'.format(i))
            self.Points.append([x, y])
        # Конечные точки
        x = tf.Variable(end_point.x, trainable=False, name='px_end')
        y = tf.Variable(end_point.y, trainable=False, name='py_end')
        self.Points.append([x, y])

        # Ускорение
        self.Accelerations = []
        for i in range(1, self.num_vars):
            self.Accelerations.append(tf.identity(self.compute_acceleration_line_direction(self.Points[i - 1], self.Points[i]),
                                                  name='acc{}'.format(i)))

        # Расчитываем скопрость, и время
        vel0 = tf.Variable(0., trainable=False, name='vel0')
        self.Velocities = [vel0]
        self.Times = []
        for i in range(1, self.num_vars):
            time, next_vel = self.compute_time_and_next_vel(self.Points[i - 1], self.Points[i],
                                                            self.Velocities[i - 1],
                                                            self.Accelerations[i - 1], is_first=i == 1)
            # # Со
            self.Velocities.append(tf.identity(next_vel, name='vel{}'.format(i)))
            self.Times.append(tf.identity(time, name='time{}'.format(i)))

        assert len(self.Points) == self.num_vars
        assert len(self.Accelerations) == self.num_vars - 1
        assert len(self.Velocities) == self.num_vars
        assert len(self.Times) == self.num_vars - 1

    def compute_acceleration_line_direction(self, point, point_next):
        """
        расчитываем гравтаци.
        :param point: первая точка
        :param point_next: вторая
        :return: укорение
        """
        x, y = point
        x_next, y_next = point_next

        # Draw a picture, and you got it
        slop = tf.atan((y_next - y) / self.dx)
        acceleration = - G * tf.cos(PI / 2. - slop)

        tf.assert_less_equal(acceleration, G)
        tf.assert_greater_equal(acceleration, -G)

        return acceleration

    def compute_time_and_next_vel(self, point, point_next, vel, acc, is_first=False):

        x, y = point
        x_next, y_next = point_next

        distance = tf.sqrt(tf.reduce_mean(tf.square(x - x_next) +
                                          tf.square(y - y_next)))


        if is_first:
            #  0.5 * acc * t^2 + vel * t - distance = 0

            time = (-vel + tf.sqrt(vel * vel + 2 * acc * distance)) / acc
            next_vel = vel + acc * time
        else:

            time_est = distance / vel
            next_vel_est = vel + acc * time_est

            average_vel = (vel + next_vel_est) / 2
            time = distance / average_vel
            next_vel = vel + acc * time

        return time, next_vel

    def loss(self):
        'минимизируем время прохождения кривой'
        loss = 0.
        for t in self.Times:
            loss += t

        return tf.reduce_mean(loss)


def BrachistochroneCurveDemo():
    """
    Demo with a plot
    """
    START_POINT = Point(x=0., y=1.)
    END_POINT = Point(x=3., y=0.)
    b_curve = BrachistochroneCurve(start_point=START_POINT, end_point=END_POINT)

    loss = b_curve.loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    training_operation = optimizer.minimize(loss)
    x_save = []
    y_save = []
    label_curve = []
    save_point = [0,50,100,299]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()

        NUM_ITERS = 300
        for i in range(NUM_ITERS):
            x_variable = [x for x, y in b_curve.Points]
            y_variable = [y for x, y in b_curve.Points]
            _, loss_val, times, X, Y, vels, Acc \
                = sess.run([training_operation, loss, b_curve.Times, x_variable, y_variable, b_curve.Velocities, b_curve.Accelerations])
            print('iters:{} loss_val:{}'.format(i, loss_val))

            if VERBOSE:
                for x, y in zip(X, Y):
                    print('Point:', x, y)
                for var in times:
                    print('time:', var)
                for var in vels:
                    print('vel:', var)
                for var in Acc:
                    print('acc:', var)

            if i in save_point:
                x_save.append(X)
                y_save.append(Y)


            plt.plot(X, Y)
            plt.title('Brachistochrone Curve')
            plt.draw()
            plt.axis('equal')
            plt.pause(0.01)
            plt.clf()

            if i == (NUM_ITERS -1):
                colors = ['r','g','m','y','b']
                for j in range(len(save_point)):
                    plt.plot(x_save[j], y_save[j], label=save_point[j], color=colors[j])

                # plt.plot(x_save[0],y_save[0],label = save_point[0],color = 'r')
                # plt.plot(x_save[1], y_save[1], label=save_point[1], color='g')
                # plt.plot(x_save[2], y_save[2], label=save_point[2], color='m')
                # plt.plot(x_save[3], y_save[3], label=save_point[3], color='y')

                plt.legend(loc='best')
                plt.title('Форма кривой в зависимости от кол-ва итераций ')




def main():
    BrachistochroneCurveDemo()




if __name__ == '__main__':
    main()
