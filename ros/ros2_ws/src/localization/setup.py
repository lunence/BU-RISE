from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=["localization", "localization.*"], exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, 
            ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='mooncyjoyli@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'localization_node = localization.localization_node:main',
            'apriltag_node = localization.apriltag_node:main'
        ],
    },
)
