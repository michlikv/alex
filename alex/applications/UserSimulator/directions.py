#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import urllib
from datetime import datetime, timedelta
import time
import json
from alex.tools.apirequest import APIRequest
from alex.utils.cache import lru_cache


class Travel(object):
    """Holder for starting and ending point (and other parameters) of travel."""

    def __init__(self, **kwargs):
        """Initializing (just filling in data).

        Accepted keys: from_city, from_stop, to_city, to_stop, vehicle, max_transfers."""
        self.from_city = kwargs['from_city']
        self.from_stop = kwargs['from_stop'] if kwargs['from_stop'] not in ['__ANY__', 'none'] else None
        self.to_city = kwargs['to_city']
        self.to_stop = kwargs['to_stop'] if kwargs['to_stop'] not in ['__ANY__', 'none'] else None
        self.vehicle = kwargs['vehicle'] if kwargs['vehicle'] not in ['__ANY__', 'none', 'dontcare'] else None
        self.max_transfers = (kwargs['max_transfers']
                              if kwargs['max_transfers'] not in  ['__ANY__', 'none', 'dontcare']
                              else None)

    def get_minimal_info(self):
        """Return minimal waypoints information
        in the form of a stringified inform() dialogue act."""
        res = []
        if self.from_city != self.to_city or (bool(self.from_stop) != bool(self.to_stop)):
            res.append("inform(from_city='%s')" % self.from_city)
        if self.from_stop is not None:
            res.append("inform(from_stop='%s')" % self.from_stop)
        if self.from_city != self.to_city or (bool(self.from_stop) != bool(self.to_stop)):
            res.append("inform(to_city='%s')" % self.to_city)
        if self.to_stop is not None:
            res.append("inform(to_stop='%s')" % self.to_stop)
        if self.vehicle is not None:
            res.append("inform(vehicle='%s')" % self.vehicle)
        if self.max_transfers is not None:
            res.append("inform(num_transfers='%s')" % str(self.max_transfers))
        return '&'.join(res)


class Directions(Travel):
    """Ancestor class for transit directions, consisting of several routes."""

    def __init__(self, **kwargs):
        if 'travel' in kwargs:
            super(Directions, self).__init__(**kwargs['travel'].__dict__)
        else:
            super(Directions, self).__init__(**kwargs)
        self.routes = []

    def __getitem__(self, index):
        return self.routes[index]

    def __len__(self):
        return len(self.routes)

    def __repr__(self):
        ret = ''
        for i, route in enumerate(self.routes, start=1):
            ret += "ROUTE " + unicode(i) + "\n" + route.__repr__() + "\n\n"
        return ret


class Route(object):
    """Ancestor class for one transit direction route."""

    def __init__(self):
        self.legs = []

    def __repr__(self):
        ret = ''
        for i, leg in enumerate(self.legs, start=1):
            ret += "LEG " + unicode(i) + "\n" + leg.__repr__() + "\n"
        return ret


class RouteLeg(object):
    """One traffic directions leg."""

    def __init__(self):
        self.steps = []

    def __repr__(self):
        return "\n".join(step.__repr__() for step in self.steps)


class RouteStep(object):
    """One transit directions step -- walking or using public transport.
    Data members:
    travel_mode -- TRANSIT / WALKING

    * For TRANSIT steps:
        departure_stop
        departure_time
        arrival_stop
        arrival_time
        headsign       -- direction of the transit line
        vehicle        -- type of the transit vehicle (tram, subway, bus)
        line_name      -- name or number of the transit line

    * For WALKING steps:
        duration       -- estimated walking duration (seconds)
    """

    MODE_TRANSIT = 'TRANSIT'
    MODE_WALKING = 'WALKING'

    # TODO this should be done somehow more clever
    STOPS_MAPPING = {'Můstek - A': 'Můstek',
                     'Můstek - B': 'Můstek',
                     'Muzeum - A': 'Muzeum',
                     'Muzeum - C': 'Muzeum',
                     'Florenc - B': 'Florenc',
                     'Florenc - C': 'Florenc'}

    def __init__(self, travel_mode):
        self.travel_mode = travel_mode

        if self.travel_mode == self.MODE_TRANSIT:
            self.departure_stop = None
            self.departure_time = None
            self.arrival_stop = None
            self.arrival_time = None
            self.headsign = None
            self.vehicle = None
            self.line_name = None

        elif self.travel_mode == self.MODE_WALKING:
            self.duration = None

    def __repr__(self):
        ret = self.travel_mode
        if self.travel_mode == self.MODE_TRANSIT:
            ret += ': ' + self.vehicle + ' ' + self.line_name + \
                    ' [^' + self.headsign + ']: ' + self.departure_stop + \
                    ' ' + str(self.departure_time) + ' -> ' + \
                    self.arrival_stop + ' ' + str(self.arrival_time)
        elif self.travel_mode == self.MODE_WALKING:
            ret += ': ' + str(self.duration / 60) + ' min, ' + \
                    ((str(self.distance) + ' m') if hasattr(self, 'distance') else '')
        return ret


class DirectionsFinderException(Exception):
    pass


class NotSupported(DirectionsFinderException):
    pass


class DirectionsFinder(object):
    """Abstract ancestor for transit direction finders."""

    def get_directions(self, travel, departure_time=None, arrival_time=None):
        """Retrieve the directions for the given travel route at the given time."""
        raise NotImplementedError()

    def get_platform(self, platform_info):
        """Retrieve the platform information for the given platform parameters."""
        raise NotSupported()


class GoogleDirections(Directions):
    """Traffic directions obtained from Google Maps API."""

    def __init__(self, input_json={}, **kwargs):
        super(GoogleDirections, self).__init__(**kwargs)
        for route in input_json['routes']:
            self.routes.append(GoogleRoute(route))


class GoogleRoute(Route):

    def __init__(self, input_json):
        super(GoogleRoute, self).__init__()
        for leg in input_json['legs']:
            self.legs.append(GoogleRouteLeg(leg))


class GoogleRouteLeg(RouteLeg):

    def __init__(self, input_json):
        super(GoogleRouteLeg, self).__init__()
        for step in input_json['steps']:
            self.steps.append(GoogleRouteLegStep(step))


class GoogleRouteLegStep(RouteStep):

    VEHICLE_TYPE_MAPPING = {'HEAVY_RAIL': 'train',
                            'Train': 'train',
                            'Long distance train': 'train'}

    def __init__(self, input_json):
        self.travel_mode = input_json['travel_mode']

        if self.travel_mode == self.MODE_TRANSIT:

            data = input_json['transit_details']
            self.departure_stop = data['departure_stop']['name']
            self.departure_time = datetime.fromtimestamp(data['departure_time']['value'])
            self.arrival_stop = data['arrival_stop']['name']
            self.arrival_time = datetime.fromtimestamp(data['arrival_time']['value'])
            self.headsign = data['headsign']
            self.line_name = data['line']['short_name']
            vehicle_type = data['line']['vehicle'].get('type', data['line']['vehicle']['name'])
            self.vehicle = self.VEHICLE_TYPE_MAPPING.get(vehicle_type, vehicle_type.lower())
            # normalize some stops' names
            self.departure_stop = self.STOPS_MAPPING.get(self.departure_stop, self.departure_stop)
            self.arrival_stop = self.STOPS_MAPPING.get(self.arrival_stop, self.arrival_stop)

        elif self.travel_mode == self.MODE_WALKING:
            self.duration = input_json['duration']['value']
            self.distance = input_json['distance']['value']

class GoogleDirectionsFinder(DirectionsFinder, APIRequest):
    """Transit direction finder using the Google Maps query engine.
       It is with 2 seconds delay due to limitations of free access to google API.
    """

    def __init__(self, cfg):
        DirectionsFinder.__init__(self)
        APIRequest.__init__(self, cfg, 'google-directions', 'Google directions query')
        self.directions_url = 'https://maps.googleapis.com/maps/api/directions/json'

    @lru_cache(maxsize=10)
    def get_directions(self, waypoints, departure_time=None, arrival_time=None):
        """Get Google maps transit directions between the given stops
        at the given time and date.

        The time/date should be given as a datetime.datetime object.
        Setting the correct date is compulsory!
        """
        #sleep 2 seconds
        time.sleep(2)
        
        data = {
            'origin': ('"zastávka %s", %s, Česká republika' %
                       (waypoints.from_stop, waypoints.from_city)).encode('utf-8'),
            'destination': ('"zastávka %s", %s, Česká republika' %
                            (waypoints.to_stop, waypoints.to_city)).encode('utf-8'),
            'region': 'cz',
            'sensor': 'false',
            'alternatives': 'true',
            'mode': 'transit',
        }
        if departure_time:
            data['departure_time'] = int(time.mktime(departure_time.timetuple()))
        elif arrival_time:
            data['arrival_time'] = int(time.mktime(arrival_time.timetuple()))

        self.system_logger.info("Google Directions request:\n" + str(data))

        page = urllib.urlopen(self.directions_url + '?' +
                              urllib.urlencode(data))
        response = json.load(page)
        self._log_response_json(response)

        directions = GoogleDirections(input_json=response, travel=waypoints)
        self.system_logger.info("Google Directions response:\n" + response['status'] + "\n" +
                                unicode(directions))
        return directions


def _todict(obj, classkey=None):
    """Convert an object graph to dictionary.
    Adapted from:
    http://stackoverflow.com/questions/1036409/recursively-convert-python-object-graph-to-dictionary .
    """
    if isinstance(obj, dict):
        for k in obj.keys():
            obj[k] = _todict(obj[k], classkey)
        return obj
    elif hasattr(obj, "__keylist__"):
        data = {key: _todict(obj[key], classkey)
                for key in obj.__keylist__
                if not callable(obj[key])}
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    elif hasattr(obj, "__dict__"):
        data = {key: _todict(value, classkey)
                for key, value in obj.__dict__.iteritems()
                if not callable(value)}
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    elif hasattr(obj, "__iter__"):
        return [_todict(v, classkey) for v in obj]
    else:
        return obj
