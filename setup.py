from setuptools import find_packages, setup

package_name = 'kalman_filter'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ebrahim',
    maintainer_email='ebrahim@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'Noise = kalman_filter.noise:main', # this is the executable file that will be created after the compilation pubExec is the name of the executable file
            'KalmanFilter = kalman_filter.2d_kalmanfilter:main', 
        ],
    },
)
