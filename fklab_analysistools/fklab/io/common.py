import re
import itertools

from collections import OrderedDict

#define voltage scaling factors
_units_scale_factor = {'V':1, 'mV':1000, 'uV':1000000}

#define validation classes for text header parameters
class ValueHandler:
    @staticmethod
    def validate(value):
        raise NotImplementedError
    
    @staticmethod
    def stringify(value):
        raise NotImplementedError

class DefaultHandler(ValueHandler):
    @staticmethod
    def validate(value):
        return value
    
    @staticmethod
    def stringify(value):
        return str(value)

class BoolHandler(DefaultHandler):
    @staticmethod
    def validate(value):
        if isinstance(value,str):
            value = value.lower() == 'true'
        else:
            value = bool(value)
            
        return value

class IntHandler(DefaultHandler):
    @staticmethod
    def validate(s):
        return int(s)

class FloatHandler(DefaultHandler):
    @staticmethod
    def validate(s):
        return float(s)

class StringEnumHandler(ValueHandler):
    def __init__(self, enum=None):
        self._enum = set([ str(k) for k in enum ])
    def validate(self,s):
        s = str(s)
        if not s in self._enum:
            raise ValueError
        
        return s
    
    stringify = validate

#class for validation of all parameters in a header
class HeaderValidator:
    def __init__(self, default=None):
        self._handlers = {}
        self._default_handler = default
        if not default is None and not isinstance(default,ValueHandler) and not issubclass(default,ValueHandler):
            raise ValueError
    
    def set_handler(self, handler, targets):
        if not isinstance(handler,ValueHandler) and not issubclass(handler,ValueHandler):
            raise ValueError
        
        if isinstance(targets,str):
            self._handlers[targets] = handler
        else:
            for k in targets:
                self._handlers[str(k)] = handler
    
    def _get_handler(self,target):
        if target in self._handlers.keys():
            return self._handlers[target]
        
        for k in self._handlers.iterkeys():
            if re.match(k,target):
                return self._handlers[k] 
        
        if self._default_handler is not None:
            return self._default_handler
        
        raise KeyError
    
    def validate(self, target, value ):
        handler = self._get_handler(target)
        return handler.validate(value)
    
    def stringify(self, target, string ):
        handler = self._get_handler(target)
        return handler.stringify(string)

#classes to represent text headers
class subheader(object):
    
    _comment_pattern = re.compile( '__comment[0-9]+' )
    _parameter_format = "{parameter} = {value}\n"
    _comment_format = "#{comment}\n"
    
    def __init__(self, parameters=None, validator=None):
        
        self._parameters = OrderedDict()
        self._validator = validator
        
        if parameters is not None:
            for k,v in parameters:
                self[k] = v #to trigger any key/value validation
    
    def __setitem__(self,key,value):
        if not isinstance(key,str):
            raise KeyError
        
        if self._validator is not None:
            value = self._validator.validate( key, value )
        
        self._parameters[key] = value
    
    def __getitem__(self,key):
        return self._parameters[key]
    
    def __len__(self):
        return len(self._parameters)
    
    def get_parameter(self,key,**kwargs):
        try:
            value = self[key]
        except KeyError:
            if kwargs.has_key('default'):
                value = kwargs['default']
            else:
                raise
        
        return value
    
    def push(self,key,value):
        if key in self._parameters.keys():
            raise KeyError
        self[key] = value
    
    def pop(self,key,*args,**kwargs):
        return self._parameters.pop(key,*args,**kwargs)
    
    def popitem(self):
        return self._parameters.popitem()
    
    def add_comment(self,comment):
        n = self._count_comments() + 1
        self.push( '__comment'+str(n), comment )
    
    def _count_comments(self):
        n = 0
        for k in self._parameters.keys():
            if self._comment_pattern.match(k):
                n+=1
        
        return n
    
    def delete_comments(self):
        for k in self._parameters.keys():
            if self._comment_pattern.match(k):
                self._parameters.pop(k)
    
    def get_comments(self):
        comments = [ v for k,v in self._parameters.iteritems() if self._comment_pattern.match(k) ]
        return comments
    
    def has_parameter(self,key):
        return self._parameters.has_key(key)
    
    def __str__(self):
        return self.serialize()
    
    def items(self):
        return self._parameters.items()
    
    def parameters(self):
        return self._parameters.keys()
    
    def values(self):
        return self._parameters.values()
    
    def iteritems(self):
        return self._parameters.iteritems()
    
    def iterparameters(self):
        return self._parameters.iterkeys()
    
    def itervalues(self):
        return self._parameters.itervalues()
    
    def serialize(self, parameter_format=None, comment_format=None):
        
        if parameter_format is None:
            parameter_format = self._parameter_format
        
        if comment_format is None:
            comment_format = self._comment_format
        
        s = ''
        for k,v in self.iteritems():
            
            if self._comment_pattern.match(k):
                s += comment_format.format(comment=v)
            else:
                if self._validator is not None:
                    value = self._validator.stringify(k,v)
                else:
                    value = str(v)
                s += parameter_format.format(parameter=k,value=value)
            
        return s

class header(object):
    
    _parameter_format = "{parameter} = {value}\n"
    _comment_format = "#{comment}\n"
    _begin_header = "--begin header--\n"
    _end_header = "--end header--\n"
    _validator = None
    
    def __init__(self, *args, **kwargs):
        #maintains list of subheaders
        self._subheaders = []
        
        self._validator = kwargs.get( 'validator', self._validator )
        
        for k in args:
            sh = subheader( k, validator=self._validator )
            if len(sh)>0:
                self._subheaders.append(sh)
    
    def push(self,key,value):
        if len(self)==0:
            self.add_subheader()
        
        self._subheaders[-1].push(key,value)
    
    def pop(self,key,*args,**kwargs):
        if len(self)==0:
            raise KeyError
        
        v = self._subheaders[-1].pop(key,*args,**kwargs)
            
        if len(self._subheaders[-1])==0:
            del self._subheaders[-1]
            
        return v
    
    def popitem(self):
        if len(self)==0:
            raise KeyError
        
        k,v = self.subheaders[-1].popitem()
        
        if len(self._subheaders[-1])==0:
            del self._subheaders[-1]
            
        return k,v
    
    def _create_subheader(self,parameters=None):
        if parameters is None:
            h = subheader(validator=self._validator)
        elif isinstance(parameters,subheader):
            h = subheader(parameters=parameters.iteritems(), validator=self._validator)
        else:
            h = subheader(parameters=parameters, validator=self._validator)
        
        return h
    
    def add_subheader(self,parameters=None):
        self._subheaders.append( self._create_subheader(parameters) )
    
    def add_comment(self,comment):
        if len(self)==0:
            self.add_subheader()
        
        self._subheaders[-1].add_comment(comment)
    
    def delete_comments(self):
        for k in self._subheaders:
            k.delete_comments()
    
    def get_comments(self):
        return [k.get_comments() for k in self._subheaders]
    
    def has_parameter(self,key):
        return any( [k.has_parameter(key) for k in self._subheaders] )
    
    def get_parameter(self,key,**kwargs):
        
        order = kwargs.get('order','normal')
        
        if order=='normal':
            for k in self._subheaders:
                if k.has_parameter(key):
                    return k[key]
        else:
            for k in reversed(self._subheaders):
                if k.has_parameter(key):
                    return k[key]
        
        if kwargs.has_key('default'):
            return kwargs['default']
        
        raise KeyError
    
    def __setitem__(self,key,value):
        if isinstance(key,str):
            #header['key']=value
            if len(self)==0:
                self.add_subheader()
            self._subheaders[0][key] = value
        else:
            h = self._create_subheader( parameters=value )
            self._subheaders[key] = h
    
    def __getitem__(self,key):
        if isinstance(key,str):
            #header['key']
            return self.get_parameter(key)
        else:
            return self._subheaders[key]
    
    def __len__(self):
        return len(self._subheaders)
    
    def __str__(self):
        return self.serialize()
    
    def serialize(self, parameter_format=None, comment_format=None, begin_header=None, end_header=None):
        if parameter_format is None:
            parameter_format = self._parameter_format
        if comment_format is None:
            comment_format = self._comment_format
        if begin_header is None:
            begin_header = self._begin_header
        if end_header is None:
            end_header = self._end_header
            
        s = begin_header
        for k in self._subheaders:
            s += k.serialize(parameter_format=parameter_format, comment_format=comment_format)
            s += comment_format.format(comment='')
        s += end_header
        
        return s
        
    def items(self):
        return [k.items() for k in self._subheaders]
    
    def parameters(self):
        return [k.parameters() for k in self._subheaders]
    
    def values(self):
        return [k.values() for k in self._subheaders]
    
    def iteritems(self):
        return itertools.chain( *[k.items() for k in self._subheaders] )
    
    def iterparameters(self):
        return itertools.chain( *[k.parameters() for k in self._subheaders] )
    
    def itervalues(self):
        return itertools.chain( *[k.values() for k in self._subheaders] )
    
    def __iter__(self):
        return self._subheaders.__iter__()
        
