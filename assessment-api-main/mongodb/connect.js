import chalk from "chalk";
import mongoose from "mongoose";

const key = process.env.MONGO_URI;
const connectDB = async () => {
    try {
        const conn = await mongoose.connect(key);
        console.log(`Database connected: ${chalk.cyan(conn.connection.host)}`);
    } catch (error) {
        console.log(`Error: ${chalk.red('Unable to connect to database')}`);
        process.exit(1);
    }
}

export default connectDB;