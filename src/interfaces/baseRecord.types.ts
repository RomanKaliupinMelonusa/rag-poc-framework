export interface BaseRecord {
    id: string;
    [key: string]: any; // Allows other properties of any type
}
