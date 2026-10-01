from setuptools import setup
import os
from glob import glob

package_name = 'crack_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pth')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Real-time crack detection for Husky UGV',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'crack_detection_node = crack_detection.crack_detection_node:main',
        ],
    },
)
