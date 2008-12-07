enum fields 
{ 
    FIELD1, 
    FIELD2 
} 
 
fields find_field(fields f) { 
    with(fields) { 
        switch(f) { 
        case FIELD1:  
	  return FIELD1;
        default: 
	  return FIELD2;
        } 
    } 
} 
 
void main() { 
  assert(find_field(fields.FIELD1) == fields.FIELD1);
  assert(find_field(fields.FIELD2) == fields.FIELD2);
} 
