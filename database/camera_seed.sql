-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Mar 04, 2022 at 09:51 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `camera_seed`
--
CREATE DATABASE `camera_seed`;
USE `camera_seed`;
-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`username`, `password`) VALUES
('admin', 'admin');

-- --------------------------------------------------------

--
-- Table structure for table `store_data`
--

CREATE TABLE `store_data` (
  `id` int(11) NOT NULL,
  `simage` varchar(50) NOT NULL,
  `name` varchar(100) NOT NULL,
  `category` varchar(10) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `store_data`
--

INSERT INTO `store_data` (`id`, `simage`, `name`, `category`) VALUES
(1, 's1.png', 'Split Pigeon Pea', 'b'),
(2, 's2.png', 'Waiting', 'e'),
(3, 's3.png', 'Chickpeas', 'a'),
(4, 's4.png', 'Chickpeas', 'a');
