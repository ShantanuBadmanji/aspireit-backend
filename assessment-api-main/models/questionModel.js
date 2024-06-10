import mongoose from "mongoose";

const questionSchema = new mongoose.Schema({
    question:{
        type: String,
        required: true
    },
    accepted_keywords:{
        type: [String],
        default: []
    },
    difficulty:{
        type: String,
        enum: ['easy', 'medium', 'hard'],
        default: 'medium'
    },
    topic:{
        type: String,
        enum: ['dsa', 'oops', 'frontend', 'backend', 'ml', 'networks', 'dbms', 'os', 'cloud'],
        default: 'dsa'
    }
},{timestamps: true});

const questionModel = mongoose.model('Question', questionSchema);

export default questionModel;