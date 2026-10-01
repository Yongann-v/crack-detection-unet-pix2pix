from setuptools import setup

package_name = 'insta360_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Publish the Insta360 ONE R (USB webcam mode) as ROS 2 image topics',
    license='MIT',
    entry_points={
        'console_scripts': [
            'insta360_publisher = insta360_ros.insta360_publisher:main',
        ],
    },
)
