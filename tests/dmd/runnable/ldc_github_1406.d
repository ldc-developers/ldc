// Tests string switch sorting, as per LDC GitHub #1406.

void fun(string command){
    switch(command) {
        case "foo_0": break;
        case "foo_1": break;
        case "foo_2": break;
        case "foo_3": break;
        case "foo_4": break;
        case "foo_5": break;
        case "foo_6": break;
        case "foo_7": break;
        case "foo_8": break;
        case "foo_9": break;
        case "foo_10": break;
        case "foo_11": break;
        case "foo_12": break;
        case "foo_13": break;
        case "foo_14": break;
        case "foo_15": break;
        case "foo_16": break;
        case "foo_17": break;
        case "foo_18": break;
        case "foo_19": break;
        case "foo_20": break;
        case "foo_21": break;
        case "foo_22": break;
        case "foo_23": break;
        case "foo_24": break;
        case "foo_25": break;
        case "foo_26": break;
        case "foo_27": break;
        case "foo_28": break;
        case "foo_29": break;
        case "foo_30": break;
        case "foo_31": break;
        case "foo_32": break;
        case "foo_33": break;
        case "foo_34": break;
        case "foo_35": break;
        case "foo_36": break;
        case "foo_37": break;
        case "foo_38": break;
        case "foo_39": break;
        default: assert(0, command);
    }
}

void main() {
    foreach (a; ["foo_19", "foo_20"]) {
        fun(a);
    }
}
