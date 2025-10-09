from setuptools import find_packages, setup

package_name = 'vision'

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
    maintainer='final-project',
    maintainer_email='karamahati@gmail.com',
    description='ROS2 package for RGB camera image viewing and processing',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'show_rgb_image = vision.show_rgb_image_node:main',
        ],
    },
)
