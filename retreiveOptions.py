from optparse import OptionParser
from os import path, makedirs

usage = "usage: %prog [options]"
parser = OptionParser(usage=usage)
parser.add_option("-s", "--singleRetreive", dest="singleRetreive", action="store_true", default=False,
                  help="Exits after a single retreive.", metavar="ST")
parser.add_option("-f", "--folder", dest="folder", default='./market-data', metavar='PATH',
                  help="Destination for storing data. (String)")

(options, args) = parser.parse_args()

if options.folder[-1] != '/':
  options.folder += '/'


if not path.exists(options.folder):
  makedirs(options.folder)